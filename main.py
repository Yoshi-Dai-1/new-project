import os
import sys
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from time import sleep
from dotenv import load_dotenv
from pathlib import Path

# 1. 検索パスにカレントディレクトリを追加（念のため）
sys.path.append(os.path.dirname(__file__))

# 2. 「フォルダ名.ファイル名」の形式でインポートする
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
(DATA_PATH / "raw/xbrl_doc").mkdir(parents=True, exist_ok=True)
(DATA_PATH / "raw/xbrl_doc_ext").mkdir(parents=True, exist_ok=True)

# 1. 書類一覧の取得（例：2024年6月の有報集中時期）
START_DATE = "2024-06-15"
END_DATE = "2024-06-30"
TARGET_SECTOR = "食料品"  # 指定業種

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

# 2. 有価証券報告書に絞り込み、指定業種でフィルタリング
print(f"業種 '{TARGET_SECTOR}' の有価証券報告書を抽出中...")
yuho_df = edinet_meta.get_yuho_df()
yuho_filtered = yuho_df.query(f"sector_label_33 == '{TARGET_SECTOR}'")
print(f"対象企業数: {len(yuho_filtered)}")

# デモ用に件数を制限（必要に応じて調整）
yuho_filtered = yuho_filtered.set_index("docID").head(10)

# 3. XBRLデータのダウンロード
print("XBRLデータをダウンロード中...")
for docid in tqdm(yuho_filtered.index):
    out_filename = str(DATA_PATH / "raw/xbrl_doc" / (docid + ".zip"))
    if not os.path.exists(out_filename):
        request_doc(api_key=API_KEY, docid=docid, out_filename_str=out_filename)
        sleep(0.5)

# 4. 財務データの抽出（大福帳の作成）
print("共通タクソノミを準備中...")
account_list = account_list_common(data_path=DATA_PATH, account_list_year="2024")

# 抽出対象のロール設定
fs_dict = {
    'BS': ["_BalanceSheet", "_ConsolidatedBalanceSheet"],
    'PL': ["_StatementOfIncome", "_ConsolidatedStatementOfIncome"],
    'report': ["_CabinetOfficeOrdinanceOnDisclosure"]
}

print("各書類から財務データを抽出中...")
fs_tbl_list = []
for docid in tqdm(yuho_filtered.index):
    try:
        fs_tbl = get_fs_tbl(
            account_list_common_obj=account_list,
            docid=docid,
            zip_file_str=str(DATA_PATH / "raw/xbrl_doc" / (docid + ".zip")),
            temp_path_str=str(DATA_PATH / "raw/xbrl_doc_ext" / docid),
            role_keyward_list=fs_dict['BS'] + fs_dict['PL'] + fs_dict['report']
        )
        # 企業名などの情報を付加
        fs_tbl = fs_tbl.assign(
            filerName=yuho_filtered.loc[docid, 'filerName'],
            sector_label_33=yuho_filtered.loc[docid, 'sector_label_33']
        )
        fs_tbl_list.append(fs_tbl)
    except Exception as e:
        print(f"エラー (docID: {docid}): {e}")

# データの統合と保存
if fs_tbl_list:
    final_df = pd.concat(fs_tbl_list)
    output_file = DATA_PATH / "financial_data.csv"
    final_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"CSV保存完了: {output_file}")
else:
    print("データが抽出されませんでした。")
