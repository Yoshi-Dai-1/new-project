import os
import sys
import pandas as pd
import shutil
import traceback
import urllib3
import requests
import signal
import json
import argparse
import calendar
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from tqdm import tqdm
from time import sleep
from dotenv import load_dotenv
from pathlib import Path
from zipfile import ZipFile
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi

# 警告の抑制
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True" # Pandera警告抑制

# 1. 検索パスにサブモジュールのルートフォルダを追加
submodule_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'edinet_xbrl_prep'))
sys.path.insert(0, submodule_root)

# 2. パッケージ形式でインポートする
from edinet_xbrl_prep.edinet_api import request_term, request_doc, edinet_response_metadata
from edinet_xbrl_prep.link_base_file_analyzer import account_list_common as OriginalALC
from edinet_xbrl_prep.fs_tbl import get_fs_tbl
import requests

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
QUALITATIVE_DIR = DATA_PATH / "qualitative"
RAW_XBRL_DIR.mkdir(parents=True, exist_ok=True)
RAW_XBRL_EXT_DIR.mkdir(parents=True, exist_ok=True)
QUALITATIVE_DIR.mkdir(parents=True, exist_ok=True)

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

PARALLEL_WORKERS = os.cpu_count() or 2
BATCH_PARALLEL_SIZE = 5

# SQLite コード削除に伴う変更
# pending_save_data = {} # 廃止
is_shutting_down = False

def signal_handler(signum, frame):
    global is_shutting_down
    if is_shutting_down: return
    is_shutting_down = True
    print(f"\n[信号 {signum} を検知] 安全な終了処理を開始します...")
    if is_shutting_down: return
    is_shutting_down = True
    print(f"\n[信号 {signum} を検知] 安全な終了処理を開始します...")
    # SQLite 保存処理は削除。Parquetはバッチ最後でまとめて処理するため、
    # ここでは強制終了せず、メインループの break を待つ
    # sys.exit(0) # main側で制御

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Pandera の警告抑制（最速で実行）
os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"

def get_last_day_of_month(date_str):
    """日付をクランプする（例：6月31日 -> 6月30日）"""
    try:
        y, m, d = map(int, date_str.split('-'))
        last_day = calendar.monthrange(y, m)[1]
        if d > last_day:
            return f"{y:04d}-{m:02d}-{last_day:02d}"
        return date_str
    except Exception:
        return date_str

# SQLite 関連関数 (get_db_path, init_db, save_sector_batch_to_db) 削除

def sanitize_zip(zip_path):
    temp_zip = zip_path.with_suffix('.tmp.zip')
    has_changes = False
    try:
        with ZipFile(zip_path, 'r') as zin:
            with ZipFile(temp_zip, 'w') as zout:
                for item in zin.infolist():
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
        if not zip_path.exists():
            return docid_str, None, None, "ZIP file not found"
            
        extract_dir = RAW_XBRL_EXT_DIR / docid_str
        
        # 役職別ロールの分類
        qualitative_roles = []
        for key in ['notes', 'report']:
            qualitative_roles.extend(FS_DICT[key])
        
        fs_df = get_fs_tbl(
            account_list_common_obj=account_list,
            docid=docid_str,
            zip_file_str=str(zip_path),
            temp_path_str=str(extract_dir),
            role_keyward_list=ALL_ROLES
        )
        
        # 数値データとテキストデータの分離
        qualitative_df = fs_df[fs_df['role'].isin(qualitative_roles)].copy()
        quantitative_df = fs_df[~fs_df['role'].isin(qualitative_roles)].copy()
        
        # 型変換 (Parquet保存時のエラー回避のため、object型は全てstrにする)
        for df in [qualitative_df, quantitative_df]:
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str)
        
        # テキストデータの保存 (Parquet)
        if not qualitative_df.empty:
            pq_path = QUALITATIVE_DIR / f"{docid_str}.parquet"
            table = pa.Table.from_pandas(qualitative_df)
            pq.write_table(table, pq_path, compression='zstd')
        
        doc_meta = {
            'docID': docid_str, 'secCode': row_dict.get('secCode'), 'filerName': row_dict.get('filerName'),
            'submitDateTime': str(row_dict['submitDateTime']), 'docDescription': row_dict.get('docDescription'),
            'periodStart': row_dict.get('periodStart'), 'periodEnd': row_dict.get('periodEnd'),
            'sector_label_33': row_dict.get('sector_label_33', 'その他')
        }
        
        if extract_dir.exists(): shutil.rmtree(extract_dir)
        return docid_str, doc_meta, quantitative_df, None
    except Exception as e:
        # FileNotFoundError 等がシリアライズエラーにならないよう str 化
        return docid_str, None, None, str(e)

def main():
    global is_shutting_down
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=os.getenv("EXTRACT_START", "2024-06-01"))
    parser.add_argument("--end", default=os.getenv("EXTRACT_END", "2024-06-30"))
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--id-list", help="Comma separated docIDs to process")
    args = parser.parse_args()

    # 日付の補正
    start_date = get_last_day_of_month(args.start)
    end_date = get_last_day_of_month(args.end)

    # list-only モード時は、全てのライブラリ出力を stderr にリダイレクトして stdout を保護
    if args.list_only:
        sys.stdout = sys.stderr
        
    res_results = request_term(api_key=API_KEY, start_date_str=start_date, end_date_str=end_date)
    edinet_meta = edinet_response_metadata(tse_sector_url="https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls", tmp_path_str=str(DATA_PATH))
    edinet_meta.set_data(res_results)
    raw_df = edinet_meta.get_metadata_pandas_df()
    
    if raw_df.empty:
        if args.list_only:
            sys.__stdout__.write("JSON_MATRIX_DATA:[]\n")
        else:
            sys.stderr.write("書類が見つかりませんでした。\n")
        return

    yuho_df = edinet_meta.get_yuho_df()
    yuho_filtered = yuho_df[yuho_df['docTypeCode'] == '120'].copy()
    
    if yuho_filtered.empty:
        if args.list_only:
            sys.__stdout__.write("JSON_MATRIX_DATA:[]\n")
        else:
            sys.stderr.write("対象書類（有報）がありません。\n")
        return

    if "docID" in yuho_filtered.columns: yuho_filtered.set_index("docID", inplace=True)
    yuho_filtered['submitYear'] = yuho_filtered['submitDateTime'].str[:4]
    yuho_filtered = yuho_filtered.sort_values(['submitYear', 'sector_label_33'])

    # リスト出力モード（マーカーをつけて出力）
    if args.list_only:
        id_sector_list = [{"id": str(idx), "sector": str(row['sector_label_33'])} for idx, row in yuho_filtered.iterrows()]
        sys.__stdout__.write("JSON_MATRIX_DATA:" + json.dumps(id_sector_list) + "\n")
        return

    # ID指定モード
    if args.id_list:
        target_ids = args.id_list.split(",")
        # 存在するIDのみに絞り込む
        valid_ids = yuho_filtered.index.intersection(target_ids)
        yuho_filtered = yuho_filtered.loc[valid_ids]

    print(f"対象書類数: {len(yuho_filtered)}")
    
    hf_token = os.getenv("HF_TOKEN")
    hf_repo = os.getenv("HF_REPO") # 例: "user/edinet-data"

    # タクソノミURLを外部ファイルから読み込み、ライブラリのハードコード値をオーバーライド
    taxonomy_urls_path = Path("taxonomy_urls.json")
    if taxonomy_urls_path.exists():
        with open(taxonomy_urls_path, "r", encoding="utf-8") as f:
            taxonomy_urls = json.load(f)
        
        # ライブラリの内部関数で使用される download_link_dict を動的に更新
        # account_list_common._download_taxonomy の中身を直接書き換えるのは難しいため
        # インスタンス生成前に辞書を保持する
        print(f"タクソノミURLリストを読み込みました ({len(taxonomy_urls)} 件)")
    else:
        taxonomy_urls = {}
        print("警告: taxonomy_urls.json が見つかりません。ライブラリのデフォルト値を使用します。")

    # ライブラリのダウンロードメソッドをモンキーパッチして、外部URLを優先するように変更
    def patched_download_taxonomy(self):
        # 外部ファイル (taxonomy_urls) にあればそれを使用、なければ内部のデフォルトを使用
        if hasattr(self, 'account_list_year') and self.account_list_year in taxonomy_urls:
            url = taxonomy_urls[self.account_list_year]
        else:
            # ライブラリ内部のハードコード値をフォールバックとして保持
            link_dict = {
                '2024':"https://www.fsa.go.jp/search/20231211/1c_Taxonomy.zip",
                "2023":"https://www.fsa.go.jp/search/20221108/1c_Taxonomy.zip",
                "2022":"https://www.fsa.go.jp/search/20211109/1c_Taxonomy.zip",
                "2021":"https://www.fsa.go.jp/search/20201110/1c_Taxonomy.zip",
                "2020":"https://www.fsa.go.jp/search/20191101/1c_Taxonomy.zip",
                "2019":"https://www.fsa.go.jp/search/20190228/1c_Taxonomy.zip",
                "2018":"https://www.fsa.go.jp/search/20180228/1c_Taxonomy.zip",
                "2017":"https://www.fsa.go.jp/search/20170228/1c.zip",
                "2016":"https://www.fsa.go.jp/search/20160314/1c.zip",
                "2015":"https://www.fsa.go.jp/search/20150310/1c.zip",
                "2014":"https://www.fsa.go.jp/search/20140310/1c.zip"
            }
            url = link_dict.get(self.account_list_year)
            
        if not url:
            raise KeyError(f"Taxonomy URL not found for year: {self.account_list_year}")
            
        r = requests.get(url, stream=True)
        with self.taxonomy_file.open(mode="wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)

    # クラスメソッドを差し替え
    OriginalALC._download_taxonomy = patched_download_taxonomy

    tasks_to_parse = []
    loaded_account_lists = {}

    for docid, row in yuho_filtered.iterrows():
        docid_str = str(docid)
        
        # タクソノミ年度の判定 (3月31日ルール)
        period_end = str(row.get('periodEnd', ''))
        try:
            py, pm, pd_day = map(int, period_end.split('-'))
            if (pm > 3) or (pm == 3 and pd_day >= 31):
                taxonomy_year = str(py)
            else:
                taxonomy_year = str(py - 1)
        except:
            taxonomy_year = row['submitYear']
            
        if taxonomy_year not in loaded_account_lists:
            print(f"共通タクソノミ準備中 ({taxonomy_year}年版)...")
            try:
                # URLリストに存在するかチェック
                if taxonomy_year not in taxonomy_urls:
                    print(f"【警告】{taxonomy_year}年版の共通タクソノミURLが taxonomy_urls.json に未登録です。")
                    print(f"この年度の書類（{docid_str}）の解析をスキップします。URLリストに以下のURLを追加してください。")
                    print(f"URL参考: https://www.fsa.go.jp/search/YYYYMMDD/1c_Taxonomy.zip")
                    continue
                
                # account_list_common のインスタンス作成
                # パッチ適用済みのクラスを使用
                obj = OriginalALC(data_path=DATA_PATH, account_list_year=taxonomy_year)
                
                loaded_account_lists[taxonomy_year] = obj
            except Exception as e:
                print(f"【エラー】タクソノミ取得失敗 ({taxonomy_year}年版): {e}")
                continue
        
        account_list = loaded_account_lists.get(taxonomy_year)
        if not account_list:
            continue
        
        # SQLiteチェックを廃止し、全てタスクに追加
        tasks_to_parse.append((docid_str, row.to_dict()))

    # 結果集約用コンテナ
    all_financial_rows = []
    processed_doc_info = [] # (docID, submitYear, sector)

    print(f"実処理対象: {len(tasks_to_parse)} 件 (並列数: {PARALLEL_WORKERS})")
    
    # バッチ実行
    with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        for i in range(0, len(tasks_to_parse), BATCH_PARALLEL_SIZE):
            if is_shutting_down: break
            batch_tasks = tasks_to_parse[i:i+BATCH_PARALLEL_SIZE]
            
            # ダウンロード
            for docid_str, row_info in batch_tasks:
                zip_path = RAW_XBRL_DIR / f"{docid_str}.zip"
                if not zip_path.exists():
                    filer_name = row_info.get('filerName', 'Unknown')
                    print(f" -> DL中: {filer_name[:15]}... ({docid_str})")
                    res_doc = request_doc(api_key=API_KEY, docid=docid_str, out_filename_str=str(zip_path))
                    if res_doc.status == "success":
                        sanitize_zip(zip_path)
                    else:
                        if zip_path.exists(): os.remove(zip_path)
                    sleep(0.5)

            # 解析実行 (account_listオブジェクトを渡す)
            futures_map = {}
            for d_id, row in batch_tasks:
                # 年度判定の再実行（tasks_to_parse作成時と同じロジック）
                period_end = str(row.get('periodEnd', ''))
                try:
                    py, pm, pd_day = map(int, period_end.split('-'))
                    if (pm > 3) or (pm == 3 and pd_day >= 31):
                        t_year = str(py)
                    else:
                        t_year = str(py - 1)
                except:
                    t_year = row['submitYear']

                acc_obj = loaded_account_lists.get(t_year)
                if acc_obj:
                    f = executor.submit(parse_worker, (d_id, row, acc_obj))
                    futures_map[f] = (d_id, row)

            for future in as_completed(futures_map):
                d_id, row = futures_map[future]
                try:
                    res_id, res_meta, res_quant_df, err = future.result()
                    if err:
                        print(f"解析エラー ({d_id}): {err}")
                    else:
                        # 数値データの蓄積
                        if res_quant_df is not None and not res_quant_df.empty:
                            all_financial_rows.append(res_quant_df)
                        
                        processed_doc_info.append({
                            "submitYear": row['submitYear'],
                            "sector": row.get('sector_label_33', 'その他'),
                            "docID": d_id
                        })
                        
                        # ZIP削除
                        z_path = RAW_XBRL_DIR / f"{d_id}.zip"
                        if z_path.exists(): os.remove(z_path)

                except Exception as e:
                    print(f"Future Result Error ({d_id}): {e}")

    # =========================================================
    # 集約と Parquet 保存・アップロード
    # =========================================================
    
    if not processed_doc_info:
        print("処理されたデータはありません。")
        return

    # DataFrame化
    info_df = pd.DataFrame(processed_doc_info)
    
    # (年度, 業種) ごとにグループ化
    groups = info_df.groupby(['submitYear', 'sector'])
    
    # HfApi インスタンス (トークンがある場合のみ)
    api = HfApi() if hf_token and hf_repo else None
    if not api:
        print("HF_TOKEN または HF_REPO が未設定のため、ローカル保存のみ行います。")

    print("\n[保存処理] Parquetファイルの作成とアップロードを開始します...")
    
    for (year, sector), group in groups:
        target_docids = set(group['docID'])
        safe_sector = str(sector).replace("/", "・").replace("\\", "・")
        
        # ファイル名の定義 (start_date, end_date を含める)
        file_base_name = f"{year}_{safe_sector}_{args.start.replace('-','')}_{args.end.replace('-','')}"
        
        # 1. 数値データ (Financial Values)
        # ==============================
        if all_financial_rows:
            target_dfs = [df for df in all_financial_rows if df['docID'].iloc[0] in target_docids]
            
            if target_dfs:
                merged_quant_df = pd.concat(target_dfs, ignore_index=True)
                
                # 型の調整
                for col in merged_quant_df.columns:
                    if merged_quant_df[col].dtype == 'object':
                        merged_quant_df[col] = merged_quant_df[col].astype(str)
                
                pq_val_path = DATA_PATH / f"{file_base_name}_values.parquet"
                table = pa.Table.from_pandas(merged_quant_df)
                pq.write_table(table, pq_val_path, compression='zstd')
                print(f"作成: {pq_val_path} ({len(merged_quant_df)} rows)")
                
                # Upload
                if api:
                    print(f" -> Uploading {pq_val_path.name}...")
                    try:
                        api.upload_file(
                            path_or_fileobj=pq_val_path,
                            path_in_repo=f"data/{year}/{safe_sector}/{pq_val_path.name}",
                            repo_id=hf_repo,
                            repo_type="dataset",
                            token=hf_token
                        )
                    except Exception as e:
                        print(f"Upload Error: {e}")
                
        # 2. テキストデータ (Qualitative Text)
        # ==================================
        text_dfs = []
        for d_id in target_docids:
            p_path = QUALITATIVE_DIR / f"{d_id}.parquet"
            if p_path.exists():
                try:
                    df_text = pd.read_parquet(p_path)
                    text_dfs.append(df_text)
                except:
                    pass
        
        if text_dfs:
            merged_text_df = pd.concat(text_dfs, ignore_index=True)
            # 型調整
            for col in merged_text_df.columns:
                if merged_text_df[col].dtype == 'object':
                    merged_text_df[col] = merged_text_df[col].astype(str)

            pq_text_path = DATA_PATH / f"{file_base_name}_text.parquet"
            table = pa.Table.from_pandas(merged_text_df)
            pq.write_table(table, pq_text_path, compression='zstd')
            print(f"作成: {pq_text_path} ({len(merged_text_df)} rows)")
            
            # Upload
            if api:
                print(f" -> Uploading {pq_text_path.name}...")
                try:
                    api.upload_file(
                        path_or_fileobj=pq_text_path,
                        path_in_repo=f"data/{year}/{safe_sector}/{pq_text_path.name}",
                        repo_id=hf_repo,
                        repo_type="dataset",
                        token=hf_token
                    )
                except Exception as e:
                    print(f"Upload Error: {e}")

    # クリーンアップ (一時ファイル)
    if QUALITATIVE_DIR.exists():
        shutil.rmtree(QUALITATIVE_DIR)
        QUALITATIVE_DIR.mkdir()
    if RAW_XBRL_DIR.exists():
        shutil.rmtree(RAW_XBRL_DIR)
        RAW_XBRL_DIR.mkdir()

    print("全ての処理が完了しました。")

if __name__ == "__main__":
    main()
