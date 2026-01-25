import os
import sys
import pandas as pd
import sqlite3
import shutil
from datetime import datetime, timedelta
from tqdm import tqdm
from time import sleep
from dotenv import load_dotenv
from pathlib import Path

# 1. 検索パスにサブモジュールのルートフォルダを追加
submodule_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'edinet_xbrl_prep'))
sys.path.insert(0, submodule_root)

# 2. パッケージ形式でインポートする
from edinet_xbrl_prep.edinet_api import request_term, request_doc, edinet_response_metadata
from edinet_xbrl_prep.link_base_file_analyzer import account_list_common
from edinet_xbrl_prep.fs_tbl import get_fs_tbl

# 環境変数の読み込み
load_dotenv()
API_KEY = os.getenv("EDINET_API_KEY")

if not API_KEY:
    print("エラー: 環境変数 EDINET_API_KEY が設定されていません。")
    exit(1)

# 設定
DATA_PATH = Path("./data")
DATA_PATH.mkdir(exist_ok=True)
RAW_XBRL_DIR = DATA_PATH / "raw/xbrl_doc"
RAW_XBRL_EXT_DIR = DATA_PATH / "raw/xbrl_doc_ext"
RAW_XBRL_DIR.mkdir(parents=True, exist_ok=True)
RAW_XBRL_EXT_DIR.mkdir(parents=True, exist_ok=True)

# 取得期間設定（全期間取得の際はここを広く設定）
# 例: 直近1ヶ月分を取得する場合
START_DATE = "2024-06-01"
END_DATE = "2024-06-30"

# 抽出対象のロール設定（全項目）
FS_DICT = {
    'BS': ["_BalanceSheet", "_ConsolidatedBalanceSheet"],
    'PL': ["_StatementOfIncome", "_ConsolidatedStatementOfIncome"],
    'CF': ["_StatementOfCashFlows", "_ConsolidatedStatementOfCashFlows"],
    'SS': ["_StatementOfChangesInEquity", "_ConsolidatedStatementOfChangesInEquity"],
    'notes': ["_Notes", "_ConsolidatedNotes"],
    'report': ["_CabinetOfficeOrdinanceOnDisclosure"]
}
ALL_ROLES = []
for roles in FS_DICT.values():
    ALL_ROLES.extend(roles)

print(f"{START_DATE} から {END_DATE} の書類一覧を取得中...")
res_results = request_term(api_key=API_KEY, start_date_str=START_DATE, end_date_str=END_DATE)

# メタデータオブジェクトの作成
# TSEの業種情報は適宜最新のものを指定
TSE_SECTOR_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
edinet_meta = edinet_response_metadata(
    tse_sector_url=TSE_SECTOR_URL,
    tmp_path_str=str(DATA_PATH)
)
edinet_meta.set_data(res_results)

# 有価証券報告書に絞り込み（全業種）
# docTypeCode=='120' (有価証券報告書), ordinanceCode=='010' (内閣府令), formCode=='030000' (第三号様式)
print("有価証券報告書を抽出中...")
yuho_df = edinet_meta.get_yuho_df()

# フィルタリング: 有価証券報告書のみを対象とする
# 訂正報告書なども含める場合は条件を緩和してください
yuho_filtered = yuho_df[yuho_df['docTypeCode'] == '120'].copy()
if yuho_filtered.empty:
    print("対象の有価証券報告書が見つかりませんでした。")
    exit(0)

# 【高速化・効率化】年度と業種でソート（同じDBへの書き込みを連続させるため）
yuho_filtered['submitYear'] = yuho_filtered['submitDateTime'].str[:4]
yuho_filtered = yuho_filtered.sort_values(['submitYear', 'sector_label_33'])

# docIDをインデックスにセット（重複排除のため）
yuho_filtered = yuho_filtered.set_index("docID")
print(f"対象書類数: {len(yuho_filtered)}")

# 共通タクソノミ準備
print("共通タクソノミを準備中...")
# 年度は適宜最新のものを使用、あるいは処理対象に合わせて動的に取得も検討
account_list = account_list_common(data_path=DATA_PATH, account_list_year="2024")

def get_db_path(year, sector_label):
    """
    年度と業種名からDBファイルのパスを生成する
    禁止文字などを置換して安全なファイル名にする
    """
    safe_sector = str(sector_label).replace("/", "・").replace("\\", "・")
    year_dir = DATA_PATH / str(year)
    year_dir.mkdir(exist_ok=True)
    return year_dir / f"{year}_{safe_sector}.db"

def init_db(conn):
    """
    DBの初期化（テーブル作成）
    """
    cursor = conn.cursor()
    # 書き込み速度向上のための設定
    cursor.execute('PRAGMA synchronous = OFF')
    cursor.execute('PRAGMA journal_mode = WAL')
    
    # 書類管理テーブル
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        docID TEXT PRIMARY KEY,
        secCode TEXT,
        filerName TEXT,
        submitDateTime TEXT,
        docDescription TEXT,
        periodStart TEXT,
        periodEnd TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # 財務データテーブル（大福帳）
    # 複合ユニーク制約で重複排除
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS financial_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        docID TEXT,
        secCode TEXT,
        filerName TEXT,
        key TEXT,
        data_str TEXT,
        period_start TEXT,
        period_end TEXT,
        instant_date TEXT,
        scenario TEXT,
        label_jp TEXT,
        label_en TEXT,
        unit TEXT,
        decimals TEXT,
        -- 他に必要なカラムがあれば追加
        UNIQUE(secCode, key, period_start, period_end, instant_date, scenario)
    )
    ''')
    
    # インデックス
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fin_docid ON financial_data(docID)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fin_seccode ON financial_data(secCode)')
    conn.commit()

def is_doc_processed(conn, docid):
    """
    そのdocIDが既に処理済み(documentsテーブルに存在)か確認
    """
    cursor = conn.cursor()
    cursor.execute('SELECT 1 FROM documents WHERE docID = ?', (docid,))
    return cursor.fetchone() is not None

def save_to_db(conn, doc_meta, fs_df):
    """
    解析結果とメタデータをDBに保存 (UPSERT)
    """
    cursor = conn.cursor()
    
    # 1. documentsテーブルへのUPSERT
    cursor.execute('''
    INSERT OR REPLACE INTO documents (docID, secCode, filerName, submitDateTime, docDescription, periodStart, periodEnd)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        doc_meta['docID'],
        doc_meta['secCode'],
        doc_meta['filerName'],
        doc_meta['submitDateTime'],
        doc_meta['docDescription'],
        doc_meta['periodStart'],
        doc_meta['periodEnd']
    ))
    
    # 2. financial_dataテーブルへのUPSERT
    data_to_insert = []
    for _, row in fs_df.iterrows():
        # 必要なカラムを抽出してリスト化
        data_to_insert.append((
            doc_meta['docID'],
            doc_meta['secCode'],
            doc_meta['filerName'],
            row.get('key'),
            str(row.get('data_str')),
            row.get('period_start'),
            row.get('period_end'),
            row.get('instant_date'),
            row.get('scenario'),
            row.get('label_jp'),
            row.get('label_en'),
            row.get('unit'),
            row.get('decimals')
        ))
    
    cursor.executemany('''
    INSERT OR REPLACE INTO financial_data (
        docID, secCode, filerName, key, data_str, 
        period_start, period_end, instant_date, scenario, 
        label_jp, label_en, unit, decimals
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', data_to_insert)
    
    conn.commit()

print("データ抽出を開始します...")

# 【高速化】DB接続を管理する変数をループの外側に配置
current_db_path = None
conn = None

# 処理ループ
for docid, row in tqdm(yuho_filtered.iterrows(), total=len(yuho_filtered)):
    try:
        # メタデータ取得
        submit_date = str(row['submitDateTime'])
        submit_year = submit_date[:4] if submit_date else "unknown"
        sector_label = row.get('sector_label_33', 'その他')
        
        # DB接続管理
        db_path = get_db_path(submit_year, sector_label)
        
        # DBパスが変わったときだけ接続を切り替える
        if db_path != current_db_path:
            if conn:
                conn.close()
            current_db_path = db_path
            conn = sqlite3.connect(db_path)
            init_db(conn)
        
        # 重複チェック（既に処理済みならスキップ）
        if is_doc_processed(conn, docid):
            continue

        # ダウンロード
        zip_path = RAW_XBRL_DIR / f"{docid}.zip"
        if not zip_path.exists():
            request_doc(api_key=API_KEY, docid=docid, out_filename_str=str(zip_path))
            sleep(0.5) # API負荷軽減
        
        # 解析処理
        extract_dir = RAW_XBRL_EXT_DIR / docid
        
        try:
            fs_tbl = get_fs_tbl(
                account_list_common_obj=account_list,
                docid=docid,
                zip_file_str=str(zip_path),
                temp_path_str=str(extract_dir),
                role_keyward_list=ALL_ROLES
            )
            
            # DB保存用のメタデータ辞書
            doc_meta = {
                'docID': docid,
                'secCode': row.get('secCode'),
                'filerName': row.get('filerName'),
                'submitDateTime': submit_date,
                'docDescription': row.get('docDescription'),
                'periodStart': row.get('periodStart'),
                'periodEnd': row.get('periodEnd')
            }
            
            # 保存実行
            save_to_db(conn, doc_meta, fs_tbl)
            
        finally:
            # クリーンアップ：一時ファイルの削除
            # ZIPファイル
            if zip_path.exists():
                os.remove(zip_path)
            # 解凍ディレクトリ
            if extract_dir.exists():
                shutil.rmtree(extract_dir)

    except Exception as e:
        print(f"エラー (docID: {docid}): {e}")
        continue

# 最後に接続を確実に閉じる
if conn:
    conn.close()

print("全ての処理が完了しました。")
