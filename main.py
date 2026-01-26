import os
import sys
import pandas as pd
import sqlite3
import shutil
import traceback
import urllib3
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from tqdm import tqdm
from time import sleep
from dotenv import load_dotenv
from pathlib import Path
from zipfile import ZipFile

# 警告の抑制
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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

# 取得期間設定（初期値：Matrixから上書きされる想定）
START_DATE = os.getenv("EXTRACT_START", "2024-06-01")
END_DATE = os.getenv("EXTRACT_END", "2024-06-30")

# 抽出対象のロール設定
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

# GitHub Actions環境ではコア数を最大限活用
PARALLEL_WORKERS = os.cpu_count() or 2
BATCH_PARALLEL_SIZE = 5 # 以前の8から、より安全な5に変更

# グローバル変数：終了信号検知時の保存用
pending_save_data = {} # db_path -> list of (meta, df)
is_shutting_down = False

def signal_handler(signum, frame):
    """終了信号を受けた際の処理：現在完了しているバッチをDBに書き込んでから終了する"""
    global is_shutting_down
    if is_shutting_down: return
    is_shutting_down = True
    print(f"\n[信号 {signum} を検知] 安全な終了処理を開始します。現在完了済みのデータを保存します...")
    for db_p, data_list in pending_save_data.items():
        save_sector_batch_to_db(db_p, data_list)
    print("保存完了。プログラムを終了します。")
    sys.exit(0)

# 信号の登録（SIGTERM, SIGINT）
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def get_db_path(year, sector_label):
    safe_sector = str(sector_label).replace("/", "・").replace("\\", "・")
    year_dir = DATA_PATH / str(year)
    year_dir.mkdir(exist_ok=True)
    return year_dir / f"{year}_{safe_sector}.db"

def init_db(conn):
    cursor = conn.cursor()
    cursor.execute('PRAGMA synchronous = OFF')
    cursor.execute('PRAGMA journal_mode = WAL')
    cursor.execute('PRAGMA cache_size = -100000') 
    cursor.execute('''CREATE TABLE IF NOT EXISTS documents (docID TEXT PRIMARY KEY, secCode TEXT, filerName TEXT, submitDateTime TEXT, docDescription TEXT, periodStart TEXT, periodEnd TEXT, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS financial_data (id INTEGER PRIMARY KEY AUTOINCREMENT, docID TEXT, secCode TEXT, filerName TEXT, key TEXT, data_str TEXT, period_start TEXT, period_end TEXT, instant_date TEXT, scenario TEXT, label_jp TEXT, label_en TEXT, unit TEXT, decimals TEXT, UNIQUE(secCode, key, period_start, period_end, instant_date, scenario))''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fin_docid ON financial_data(docID)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fin_seccode ON financial_data(secCode)')
    conn.commit()

def save_sector_batch_to_db(db_path, batch_data):
    if not batch_data: return
    conn = sqlite3.connect(db_path)
    init_db(conn)
    cursor = conn.cursor()
    try:
        docs, fin_records = [], []
        for doc_meta, fs_df in batch_data:
            docs.append((doc_meta['docID'], doc_meta['secCode'], doc_meta['filerName'], doc_meta['submitDateTime'], doc_meta['docDescription'], doc_meta['periodStart'], doc_meta['periodEnd']))
            for _, row in fs_df.iterrows():
                fin_records.append((doc_meta['docID'], doc_meta['secCode'], doc_meta['filerName'], row.get('key'), str(row.get('data_str')), row.get('period_start'), row.get('period_end'), row.get('instant_date'), row.get('scenario'), row.get('label_jp'), row.get('label_en'), row.get('unit'), row.get('decimals')))
        cursor.executemany('INSERT OR REPLACE INTO documents (docID, secCode, filerName, submitDateTime, docDescription, periodStart, periodEnd) VALUES (?, ?, ?, ?, ?, ?, ?)', docs)
        cursor.executemany('INSERT OR REPLACE INTO financial_data (docID, secCode, filerName, key, data_str, period_start, period_end, instant_date, scenario, label_jp, label_en, unit, decimals) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)', fin_records)
        conn.commit()
    except Exception as e:
        print(f"DB保存エラー: {e}")
    finally:
        conn.close()

def sanitize_zip(zip_path):
    """解析に不要なファイルをZIPから削除（ixbrl/画像を間引く）"""
    temp_zip = zip_path.with_suffix('.tmp.zip')
    has_changes = False
    try:
        with ZipFile(zip_path, 'r') as zin:
            with ZipFile(temp_zip, 'w') as zout:
                for item in zin.infolist():
                    # 財務構造データに必要な拡張子のみ残す (.xbrl, .xml, .xsd)
                    # .htm (ixbrl), .png, .jpg (画像), .pdf などはスキップ
                    if item.filename.lower().endswith(('.xbrl', '.xml', '.xsd')):
                        zout.writestr(item, zin.read(item.filename))
                    else:
                        has_changes = True
        if has_changes:
            os.replace(temp_zip, zip_path)
        else:
            if temp_zip.exists(): os.remove(temp_zip)
    except Exception as e:
        if temp_zip.exists(): os.remove(temp_zip)
        print(f"ZIP軽量化失敗: {e}")

def parse_worker(task):
    docid_str, row_dict, account_list = task
    try:
        zip_path = RAW_XBRL_DIR / f"{docid_str}.zip"
        extract_dir = RAW_XBRL_EXT_DIR / docid_str
        
        # 構造化データの抽出実行
        fs_df = get_fs_tbl(
            account_list_common_obj=account_list,
            docid=docid_str,
            zip_file_str=str(zip_path),
            temp_path_str=str(extract_dir),
            role_keyward_list=ALL_ROLES
        )
        
        doc_meta = {
            'docID': docid_str, 'secCode': row_dict.get('secCode'), 'filerName': row_dict.get('filerName'),
            'submitDateTime': str(row_dict['submitDateTime']), 'docDescription': row_dict.get('docDescription'),
            'periodStart': row_dict.get('periodStart'), 'periodEnd': row_dict.get('periodEnd'),
            'sector_label_33': row_dict.get('sector_label_33', 'その他')
        }
        
        if extract_dir.exists(): shutil.rmtree(extract_dir)
        return docid_str, doc_meta, fs_df, None
    except Exception as e:
        return docid_str, None, None, str(e)

def main():
    global pending_save_data
    print(f"[{START_DATE} 〜 {END_DATE}] 抽出ミッション開始")
    res_results = request_term(api_key=API_KEY, start_date_str=START_DATE, end_date_str=END_DATE)
    
    edinet_meta = edinet_response_metadata(tse_sector_url="https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls", tmp_path_str=str(DATA_PATH))
    edinet_meta.set_data(res_results)
    raw_df = edinet_meta.get_metadata_pandas_df()
    if raw_df.empty:
        print("指定期間内に書類が見つかりませんでした。")
        return

    yuho_df = edinet_meta.get_yuho_df()
    yuho_filtered = yuho_df[yuho_df['docTypeCode'] == '120'].copy()
    if yuho_filtered.empty:
        print("対象書類がありません。")
        return

    if "docID" in yuho_filtered.columns: yuho_filtered.set_index("docID", inplace=True)
    yuho_filtered['submitYear'] = yuho_filtered['submitDateTime'].str[:4]
    yuho_filtered = yuho_filtered.sort_values(['submitYear', 'sector_label_33'])
    
    print(f"対象書類数: {len(yuho_filtered)}")
    print("共通タクソノミ準備中...")
    account_list = account_list_common(data_path=DATA_PATH, account_list_year="2024")

    tasks_to_parse = []
    current_db_path, processed_docids = None, set()

    for docid, row in yuho_filtered.iterrows():
        docid_str = str(docid)
        db_p = get_db_path(row['submitYear'], row.get('sector_label_33', 'その他'))
        
        if current_db_path != db_p:
            processed_docids = set()
            if db_p.exists():
                conn_check = sqlite3.connect(db_p)
                if conn_check.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='documents'").fetchone():
                    processed_docids = {str(r[0]) for r in conn_check.execute('SELECT docID FROM documents').fetchall()}
                conn_check.close()
        current_db_path = db_p
        
        if docid_str not in processed_docids:
            tasks_to_parse.append((docid_str, row.to_dict()))

    print(f"実処理対象: {len(tasks_to_parse)} 件 (並列数: {PARALLEL_WORKERS})")
    pbar_total = tqdm(total=len(tasks_to_parse), desc="進捗")

    with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        for i in range(0, len(tasks_to_parse), BATCH_PARALLEL_SIZE):
            if is_shutting_down: break
            batch_tasks = tasks_to_parse[i:i+BATCH_PARALLEL_SIZE]
            
            # 1. 順次ダウンロード & 軽量化
            for docid_str, row_info in batch_tasks:
                zip_path = RAW_XBRL_DIR / f"{docid_str}.zip"
                if not zip_path.exists():
                    filer_name = row_info.get('filerName', 'Unknown')
                    print(f" -> DL中: {filer_name[:15]}... ({docid_str})")
                    res_doc = request_doc(api_key=API_KEY, docid=docid_str, out_filename_str=str(zip_path))
                    if res_doc.status == "success":
                        sanitize_zip(zip_path) # 解析前に不要ファイルを間引く
                    else:
                        if zip_path.exists(): os.remove(zip_path)
                    sleep(0.5)

            # 2. 並列解析
            futures = [executor.submit(parse_worker, (d_id, row, account_list)) for d_id, row in batch_tasks]
            pending_save_data = {} # バッチごとにリセット
            
            for future in as_completed(futures):
                d_id, meta, df, err = future.result()
                pbar_total.update(1)
                if not err:
                    db_p = get_db_path(meta['submitDateTime'][:4], meta['sector_label_33'])
                    if db_p not in pending_save_data: pending_save_data[db_p] = []
                    pending_save_data[db_p].append((meta, df))
                    z_path = RAW_XBRL_DIR / f"{d_id}.zip"
                    if z_path.exists(): os.remove(z_path)
                else:
                    print(f"    !!! 解析失敗 ({d_id}): {err}")

            # 3. バッチ保存
            for db_p, data_list in pending_save_data.items():
                save_sector_batch_to_db(db_p, data_list)
            pending_save_data = {} # 正常保存完了

    pbar_total.close()
    print("ミッション完了")

if __name__ == "__main__":
    main()
