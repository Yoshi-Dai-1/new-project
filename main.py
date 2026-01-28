import argparse
import json
import os
import shutil
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

# モジュールのインポート
from catalog_manager import CatalogManager
from edinet_engine import EdinetEngine
from history_engine import HistoryEngine
from master_merger import MasterMerger

# サブモジュールのインポート設定
submodule_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "edinet_xbrl_prep"))
sys.path.insert(0, submodule_root)

# サブモジュールからのインポート (パス設定後に読み込み)
from edinet_xbrl_prep.fs_tbl import get_fs_tbl  # noqa: E402

# 設定
DATA_PATH = Path("data")
RAW_BASE_DIR = DATA_PATH / "raw"
TEMP_DIR = DATA_PATH / "temp"
PARALLEL_WORKERS = 4
BATCH_PARALLEL_SIZE = 8

is_shutting_down = False


def signal_handler(sig, frame):
    global is_shutting_down
    logger.warning("中断信号を受信しました。シャットダウンしています...")
    is_shutting_down = True


signal.signal(signal.SIGINT, signal_handler)


def parse_worker(args):
    """並列実行される解析ワーカー"""
    docid, row_dict, account_list, zip_path = args
    extract_dir = TEMP_DIR / docid
    try:
        roles = ["BS", "PL", "CF", "SS"]
        df = get_fs_tbl(
            account_list_common_obj=account_list,
            docid=docid,
            zip_file_str=str(zip_path),
            temp_path_str=str(extract_dir),
            role_keyward_list=roles,
        )

        if df is not None and not df.empty:
            df["docid"] = docid
            df["submitDateTime"] = row_dict.get("submitDateTime", "")
            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].astype(str)
            return docid, df, None
        return docid, None, None
    except Exception as e:
        return docid, None, str(e)
    finally:
        if extract_dir.exists():
            shutil.rmtree(extract_dir)


def main():
    parser = argparse.ArgumentParser(description="Integrated Disclosure Data Lakehouse 2.0")
    parser.add_argument("--start", type=str, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="YYYY-MM-DD")
    parser.add_argument("--id-list", type=str, help="Comma separated docIDs", default=None)
    parser.add_argument("--list-only", action="store_true", help="Output metadata as JSON for GHA matrix")
    args = parser.parse_args()

    api_key = os.getenv("EDINET_API_KEY")
    hf_token = os.getenv("HF_TOKEN")
    hf_repo = os.getenv("HF_REPO")

    if not api_key:
        logger.critical("EDINET_API_KEY が設定されていません。")
        return

    if not args.start:
        args.start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    if not args.end:
        args.end = datetime.now().strftime("%Y-%m-%d")

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger.add(log_dir / "pipeline_{time}.log", rotation="10 MB", level="INFO")

    edinet = EdinetEngine(api_key, DATA_PATH)
    catalog = CatalogManager(hf_repo, hf_token, DATA_PATH)
    merger = MasterMerger(hf_repo, hf_token, DATA_PATH)
    history = HistoryEngine(DATA_PATH)

    # 1. 市場マスタと履歴の更新 (リスト出力時も業種判定のために必要)
    if not args.id_list:
        try:
            jpx_master = history.fetch_jpx_master()
            listing_events = history.generate_listing_events(catalog.master_df, jpx_master)
            catalog.update_stocks_master(jpx_master)
            catalog.update_listing_history(listing_events)

            nk_list = history.fetch_nikkei_225_events()
            index_events = history.generate_index_events("Nikkei225", pd.DataFrame(), nk_list)
            catalog.update_index_history(index_events)
            logger.info("市場マスタ・履歴の更新が完了しました。")
        except Exception as e:
            logger.error(f"市場履歴更新中にエラーが発生しました: {e}")

    # 2. メタデータ取得
    all_meta = edinet.fetch_metadata(args.start, args.end)
    if not all_meta:
        if args.list_only:
            print("JSON_MATRIX_DATA: []")
        return

    # 3. GHAマトリックス用出力
    if args.list_only:
        matrix_data = []
        for row in all_meta:
            docid = row["docID"]
            if catalog.is_processed(docid):
                continue
            # マスタ更新後なので適切な業種が取得可能
            matrix_data.append({"id": docid, "sector": catalog.get_sector(row.get("secCode", "")[:4])})
        print(f"JSON_MATRIX_DATA: {json.dumps(matrix_data)}")
        return

    logger.info("=== Data Lakehouse 2.0 実行開始 ===")

    # 4. 処理対象の選定
    tasks = []
    new_catalog_records = []
    loaded_acc = {}

    target_ids = args.id_list.split(",") if args.id_list else None

    for row in all_meta:
        docid = row["docID"]
        if target_ids and docid not in target_ids:
            continue
        if not target_ids and catalog.is_processed(docid):
            continue

        y, m = row["submitDateTime"][:4], row["submitDateTime"][5:7]
        raw_dir = RAW_BASE_DIR / "edinet" / y / m
        raw_zip = raw_dir / f"{docid}.zip"
        raw_pdf = raw_dir / f"{docid}.pdf"

        zip_ok = edinet.download_doc(docid, raw_zip, 1) if row.get("xbrlFlag") == "1" else False
        pdf_ok = edinet.download_doc(docid, raw_pdf, 2) if row.get("pdfFlag") == "1" else False

        new_catalog_records.append(
            {
                "doc_id": docid,
                "source": "EDINET",
                "code": row.get("secCode", "")[:4],
                "edinet_code": row.get("edinetCode", ""),
                "company_name": row.get("filerName", "Unknown"),
                "doc_type": row.get("docTypeCode", ""),
                "title": row.get("docDescription", ""),
                "submit_at": row.get("submitDateTime", ""),
                "raw_zip_path": f"raw/edinet/{y}/{m}/{docid}.zip" if zip_ok else "",
                "pdf_path": f"raw/edinet/{y}/{m}/{docid}.pdf" if pdf_ok else "",
                "processed_status": "success",
            }
        )

        if row.get("docTypeCode") in ["120", "130"] and zip_ok:
            ty = row["submitDateTime"][:4]
            if ty not in loaded_acc:
                loaded_acc[ty] = edinet.get_account_list(ty)
            if loaded_acc[ty]:
                tasks.append((docid, row, loaded_acc[ty], raw_zip))

        # 50件ごとに進捗を報告
        processed_count = len(new_catalog_records)
        if processed_count % 50 == 0:
            logger.info(f"ダウンロード進捗: {processed_count} / {len(all_meta)} 件完了")

    if new_catalog_records:
        catalog.update_catalog(new_catalog_records)

    # 5. 並列解析
    all_quant_dfs = []
    processed_infos = []

    if tasks:
        logger.info(f"解析対象: {len(tasks)} 件")
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            for i in range(0, len(tasks), BATCH_PARALLEL_SIZE):
                if is_shutting_down:
                    break
                batch = tasks[i : i + BATCH_PARALLEL_SIZE]
                futures = [executor.submit(parse_worker, t) for t in batch]

                for f in as_completed(futures):
                    did, quant_df, err = f.result()
                    if err:
                        logger.error(f"解析失敗: {did} - {err}")
                    elif quant_df is not None:
                        all_quant_dfs.append(quant_df)
                        meta_row = next(m for m in all_meta if m["docID"] == did)
                        processed_infos.append(
                            {"docID": did, "sector": catalog.get_sector(meta_row.get("secCode", "")[:4])}
                        )

                # 10件（1バブル）ごとに進捗を報告
                done_count = i + len(batch)
                logger.info(
                    f"解析進捗: {min(done_count, len(tasks))} / {len(tasks)} 件完了 (成功累積: {len(all_quant_dfs)})"
                )

    # 6. マスターマージ
    if all_quant_dfs:
        logger.info("マスターマージを開始します...")
        full_quant_df = pd.concat(all_quant_dfs, ignore_index=True)
        info_df = pd.DataFrame(processed_infos)
        for sector in info_df["sector"].unique():
            sec_docids = info_df[info_df["sector"] == sector]["docID"].tolist()
            sec_quant = full_quant_df[full_quant_df["docid"].isin(sec_docids)]
            merger.merge_and_upload(sector, "financial_values", sec_quant)

    logger.success("=== パイプライン完了 ===")


if __name__ == "__main__":
    main()
