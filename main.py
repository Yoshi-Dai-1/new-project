import argparse
import json
import os
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

# サブモジュールからのインポート (動的パス追加を廃止し、正規の階層で指定)
from edinet_xbrl_prep.edinet_xbrl_prep.fs_tbl import get_fs_tbl
from history_engine import HistoryEngine
from master_merger import MasterMerger

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
    """並列処理用ワーカー関数"""
    docid, row, acc_obj, raw_zip, role_kws, task_type = args
    # 【修正】タスクタイプごとに個別の作業ディレクトリを作成し、競合（Race Condition）を回避
    extract_dir = TEMP_DIR / f"{docid}_{task_type}"
    try:
        if acc_obj is None:
            return docid, None, "Account list not loaded"

        logger.debug(f"解析開始: {docid} (Path: {raw_zip})")

        # 開発者ブログ推奨の get_fs_tbl を呼び出し
        df = get_fs_tbl(
            account_list_common_obj=acc_obj,
            docid=docid,
            zip_file_str=str(raw_zip),
            temp_path_str=str(extract_dir),
            role_keyward_list=role_kws,
        )

        if df is not None and not df.empty:
            df["docid"] = docid
            df["submitDateTime"] = row.get("submitDateTime", "")
            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].astype(str)
            logger.debug(f"解析成功: {docid} ({task_type}) | 抽出レコード数: {len(df)}")
            return docid, df, None, task_type

        msg = "No objects to concatenate" if (df is None or df.empty) else "Empty Results"
        return docid, None, msg, task_type

    except Exception as e:
        import traceback

        err_detail = traceback.format_exc()
        logger.error(f"解析例外: {docid} ({task_type})\n{err_detail}")
        return docid, None, f"{str(e)}", task_type
    finally:
        if extract_dir.exists():
            import shutil

            shutil.rmtree(extract_dir)


def main():
    # 原因追跡のため、受け取った生の引数をログに出力（デバッグ用）
    logger.debug(f"起動引数: {sys.argv}")

    parser = argparse.ArgumentParser(description="Integrated Disclosure Data Lakehouse 2.0")
    parser.add_argument("--start", type=str, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="YYYY-MM-DD")
    # ハイフン形式とアンダースコア形式の両方を受け入れ、destを統一
    parser.add_argument("--id-list", "--id_list", type=str, dest="id_list", help="Comma separated docIDs", default=None)
    parser.add_argument("--list-only", action="store_true", help="Output metadata as JSON for GHA matrix")

    try:
        args = parser.parse_args()
    except SystemExit as e:
        if e.code != 0:
            logger.error(f"引数解析エラー (exit code {e.code}): 渡された引数が不正です。 sys.argv={sys.argv}")
        raise e

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

    # 1. 市場マスタと履歴の更新 (常時実行: 履歴の断絶を防ぐため)
    try:
        jpx_master = history.fetch_jpx_master()
        if not jpx_master.empty:
            # 過去の履歴を取得して再上場判定に使用
            old_listing = catalog.get_listing_history()
            listing_events = history.generate_listing_events(catalog.master_df, jpx_master, old_listing)

            # カタログ内の master_df を最新化
            catalog.update_stocks_master(jpx_master)
            catalog.update_listing_history(listing_events)

            nk_list = history.fetch_nikkei_225_events()
            old_index = catalog.get_index_history()
            index_events = history.generate_index_events("Nikkei225", pd.DataFrame(), nk_list, old_index)
            catalog.update_index_history(index_events)

            logger.info("市場マスタ・履歴の更新が完了しました。")
    except Exception as e:
        logger.error(f"市場履歴更新中にエラーが発生しました: {e}")

    # 1. メタデータ取得
    all_meta = edinet.fetch_metadata(args.start, args.end)
    if not all_meta:
        if args.list_only:
            print("JSON_MATRIX_DATA: []")
        return

    # 【投資特化】証券コードがない（非上場企業）を即座に除外
    initial_count = len(all_meta)

    # フィルタリング理由の追跡ログ
    filtered_meta = []
    skipped_reasons = {"no_sec_code": 0, "invalid_length": 0}

    for row in all_meta:
        sec_code = str(row.get("secCode", "")).strip()
        if not sec_code:
            skipped_reasons["no_sec_code"] += 1
            continue
        if len(sec_code) < 5:
            skipped_reasons["invalid_length"] += 1
            # 56件の書類漏れなどの追跡用
            logger.debug(f"書類スキップ (コード短縮): {row.get('docID')} - {sec_code}")
            continue
        filtered_meta.append(row)

    all_meta = filtered_meta
    if initial_count > len(all_meta) and not args.id_list:
        logger.info(
            f"フィルタリング結果: 初期 {initial_count} 件 -> 保持 {len(all_meta)} 件 "
            f"(証券コードなし: {skipped_reasons['no_sec_code']} 件, "
            f"コード不正/短縮: {skipped_reasons['invalid_length']} 件)"
        )

    # 2. GHAマトリックス用出力
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
    # 【カタログ整合性】ダウンロード直後ではなく、解析結果に基づき登録するよう設計変更
    potential_catalog_records = {}  # docid -> record_base
    loaded_acc = {}
    skipped_types = {}  # 新たに追加

    target_ids = args.id_list.split(",") if args.id_list else None

    # 解析タスクの追加 (XBRL がある Yuho/Shihanki のみ)
    # 開発者ブログの指定 + 追加ロール (CF, SS, Notes)
    fs_dict = {
        "BS": ["_BalanceSheet", "_ConsolidatedBalanceSheet"],
        "PL": ["_StatementOfIncome", "_ConsolidatedStatementOfIncome"],
        "CF": ["_StatementOfCashFlows", "_ConsolidatedStatementOfCashFlows"],
        "SS": ["_StatementOfChangesInEquity", "_ConsolidatedStatementOfChangesInEquity"],
        "notes": ["_Notes", "_ConsolidatedNotes"],
        "report": ["_CabinetOfficeOrdinanceOnDisclosure"],
    }

    # ロール定義の分離
    quant_roles = fs_dict["BS"] + fs_dict["PL"] + fs_dict["CF"] + fs_dict["SS"] + fs_dict["report"]
    text_roles = fs_dict["notes"]

    for row in all_meta:
        docid = row["docID"]
        title = row.get("docDescription", "名称不明")

        if target_ids and docid not in target_ids:
            continue
        if not target_ids and catalog.is_processed(docid):
            continue

        y, m = row["submitDateTime"][:4], row["submitDateTime"][5:7]
        raw_dir = RAW_BASE_DIR / "edinet" / y / m
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_zip = raw_dir / f"{docid}.zip"
        raw_pdf = raw_dir / f"{docid}.pdf"

        # ダウンロード実行 (フラグに基づき正確に試行)
        has_xbrl = row.get("xbrlFlag") == "1"
        has_pdf = row.get("pdfFlag") == "1"

        zip_ok = False
        if has_xbrl:
            zip_ok = edinet.download_doc(docid, raw_zip, 1)
            if zip_ok:
                # RAWアップロードの実行 (Hugging Face 上の raw/ 層を構築)
                repo_zip_path = f"raw/edinet/{y}/{m}/{docid}.zip"
                if not catalog.upload_raw(raw_zip, repo_zip_path):
                    logger.error(f"RAW(XBRL)アップロード失敗: {docid}")
                    zip_ok = False
            else:
                logger.error(f"XBRLダウンロード失敗: {docid} | {title}")

        pdf_ok = False
        if has_pdf:
            pdf_ok = edinet.download_doc(docid, raw_pdf, 2)
            if pdf_ok:
                # RAWアップロードの実行 (Hugging Face 上の raw/ 層を構築)
                repo_pdf_path = f"raw/edinet/{y}/{m}/{docid}.pdf"
                if not catalog.upload_raw(raw_pdf, repo_pdf_path):
                    logger.error(f"RAW(PDF)アップロード失敗: {docid}")
                    pdf_ok = False
            else:
                logger.error(f"PDFダウンロード失敗: {docid} | {title}")

        file_status = []
        if has_xbrl:
            file_status.append("XBRLあり" if zip_ok else "XBRL(DL失敗)")
        if has_pdf:
            file_status.append("PDFあり" if pdf_ok else "PDF(DL失敗)")
        status_str = " + ".join(file_status) if file_status else "ファイルなし"

        # カタログ情報のベースを保持
        record = {
            "doc_id": docid,
            "source": "EDINET",
            "code": row.get("secCode", "")[:4],
            "edinet_code": row.get("edinetCode", ""),
            "company_name": row.get("filerName", "Unknown"),
            "doc_type": row.get("docTypeCode", ""),
            "title": title,
            "submit_at": row.get("submitDateTime", ""),
            "raw_zip_path": f"raw/edinet/{y}/{m}/{docid}.zip" if zip_ok else "",
            "pdf_path": f"raw/edinet/{y}/{m}/{docid}.pdf" if pdf_ok else "",
            "processed_status": "success" if (zip_ok or pdf_ok) else "failure",
        }
        potential_catalog_records[docid] = record

        # 解析タスクの判定 (有価証券報告書 120 + 府令 010 + 様式 030000)
        dtc = row.get("docTypeCode")
        ord_c = row.get("ordinanceCode")
        form_c = row.get("formCode")
        # 開発者ブログの指定: 種別=120, 政令=010, 様式=030000
        is_yuho = dtc == "120" and ord_c == "010" and form_c == "030000"

        if is_yuho and zip_ok:
            ty = row["submitDateTime"][:4]
            if ty not in loaded_acc:
                loaded_acc[ty] = edinet.get_account_list(ty)
            if loaded_acc[ty]:
                # 数値データのタスク
                tasks.append((docid, row, loaded_acc[ty], raw_zip, quant_roles, "financial_values"))
                # テキストデータのタスク
                tasks.append((docid, row, loaded_acc[ty], raw_zip, text_roles, "qualitative_text"))

                logger.info(f"【解析対象】: {docid} | {title} | {status_str}")
        else:
            reason = "非解析対象（有報以外）" if not is_yuho else "XBRLなし"
            logger.info(f"【スキップ】: {docid} | {title} | {status_str} | 理由: {reason}")
            skipped_types[dtc] = skipped_types.get(dtc, 0) + 1
            # 解析対象外でもファイルがあればカタログに積む
            new_catalog_records.append(record)

        # 50件ごとに進捗を報告
        processed_count = len(potential_catalog_records)
        if not args.id_list and processed_count % 50 == 0:
            logger.info(f"ダウンロード進捗: {processed_count} / {len(all_meta)} 件完了")

    # 解析対象外の書類（PDFのみ、種別違い等）をこのタイミングで一度カタログ保存
    if new_catalog_records:
        logger.info(f"解析対象外の書類 {len(new_catalog_records)} 件をカタログに登録します。")
        catalog.update_catalog(new_catalog_records)
        new_catalog_records = []

    if skipped_types:
        logger.info(f"解析スキップ内訳 (Yuho以外): {skipped_types}")

    # 5. 並列解析
    all_quant_dfs = []
    all_text_dfs = []  # テキスト用
    processed_infos = []

    if tasks:
        logger.info(f"解析対象: {len(tasks) // 2} 書類 (Task数: {len(tasks)})")

        # 解析スキップの内訳に有報(120)などが混ざっていないか、最終確認用ログ
        check_yuho_in_skip = [k for k in skipped_types.keys() if k in ["120", "121"]]
        if check_yuho_in_skip:
            logger.warning(f"注意: 解析対象であるはずの種別がスキップされています: {check_yuho_in_skip}")

        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            for i in range(0, len(tasks), BATCH_PARALLEL_SIZE):
                if is_shutting_down:
                    break
                batch = tasks[i : i + BATCH_PARALLEL_SIZE]
                futures = [executor.submit(parse_worker, t) for t in batch]

                for f in as_completed(futures):
                    did, res_df, err, t_type = f.result()

                    if err:
                        # テキスト抽出の失敗(No objects...)は頻繁にあるため、Warningレベルに留める場合もあるが
                        # ここでは一律ログ出力。ただしCatalogへの記録は "どちらも失敗" の場合のみ考慮が必要だが
                        # 現状はシンプルにエラーログのみ。
                        # "No objects to concatenate" は正常な空振りの可能性が高い。
                        level = logger.warning if "No objects" in err else logger.error
                        level(f"解析結果({t_type}): {did} - {err}")

                        # データなしフラグの処理は構造上複雑になるため、
                        # 「ダウンロード成功ならカタログにはsuccessとして残る」という現状を維持。
                        # 解析の成否は master データへの結合有無で決まる。

                    elif res_df is not None:
                        if t_type == "financial_values":
                            all_quant_dfs.append(res_df)
                        elif t_type == "qualitative_text":
                            all_text_dfs.append(res_df)

                        # processed_infos はセクター判定用。重複を防ぐため docid ごとに一度だけ追加したいが
                        # リスト内包表記で docid を抽出するので重複しても問題ない、または
                        # set で管理する手もある。ここでは単純に追加。
                        meta_row = next(m for m in all_meta if m["docID"] == did)
                        processed_infos.append(
                            {"docID": did, "sector": catalog.get_sector(meta_row.get("secCode", "")[:4])}
                        )

                # バッチごとに登録可能な未定記録を登録
                if new_catalog_records:
                    catalog.update_catalog(new_catalog_records)
                    new_catalog_records = []

                done_count = i + len(batch)
                logger.info(
                    f"解析進捗: {min(done_count, len(tasks))} / {len(tasks)} tasks 完了 "
                    f"(Quant: {len(all_quant_dfs)}, Text: {len(all_text_dfs)})"
                )
    new_catalog_records = []

    # 6. マスターマージ & カタログ確定
    all_success = True
    info_df = pd.DataFrame(processed_infos)

    # セクターリスト (重複排除)
    sectors = info_df["sector"].unique() if not info_df.empty else []

    if all_quant_dfs:
        logger.info("数値データ(financial_values)のマージを開始します...")
        full_quant_df = pd.concat(all_quant_dfs, ignore_index=True)
        for sector in sectors:
            sec_docids = info_df[info_df["sector"] == sector]["docID"].tolist()
            sec_quant = full_quant_df[full_quant_df["docid"].isin(sec_docids)]
            if not merger.merge_and_upload(sector, "financial_values", sec_quant):
                all_success = False

    if all_text_dfs:
        logger.info("テキストデータ(qualitative_text)のマージを開始します...")
        full_text_df = pd.concat(all_text_dfs, ignore_index=True)
        for sector in sectors:
            sec_docids = info_df[info_df["sector"] == sector]["docID"].tolist()
            sec_text = full_text_df[full_text_df["docid"].isin(sec_docids)]
            if not merger.merge_and_upload(sector, "qualitative_text", sec_text):
                all_success = False

    # カタログ更新（全データ処理後）
    # アップロードに成功したdocid (Quant/Text問わず、何らかのデータが保存できたもの)
    # 厳密な判定は難しいが、ここではmergerの戻り値ベースで判定
    if all_quant_dfs or all_text_dfs:
        # ダウンロード済みのものは potential_catalog_records にある
        # ここでは「データ保存まで完遂した」という意味での更新は不要かもしれない
        # （ダウンロード時に processed_status=success でレコード作成済みで、update_catalogも呼ばれているため）
        # ただし、main.py の設計上、最後にまとめて update_catalog を呼んでいた箇所。
        # 360行目で都度呼んでいるので、ここは「最終的な完了ログ」だけで良い可能性。
        pass

    if all_success:
        logger.success("=== パイプライン完了 (正常終了) ===")
    else:
        logger.error("=== パイプライン完了 (一部のアップロードに失敗しました) ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
