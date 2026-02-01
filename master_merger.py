import time
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from loguru import logger


class MasterMerger:
    def __init__(self, hf_repo: str, hf_token: str, data_path: Path):
        self.hf_repo = hf_repo
        self.hf_token = hf_token
        self.data_path = data_path
        self.api = HfApi() if hf_repo and hf_token else None

    def merge_and_upload(
        self,
        sector: str,
        master_type: str,
        new_data: pd.DataFrame,
        worker_mode: bool = False,
        catalog_manager=None,
        run_id: str = None,
        chunk_id: str = None,
    ) -> bool:
        """業種別にParquetをロード・結合・アップロード"""
        if new_data.empty:
            return True

        safe_sector = str(sector).replace("/", "・").replace("\\", "・")

        # 【修正】Workerモードならデルタ保存のみ行う
        if worker_mode:
            if not catalog_manager or not run_id or not chunk_id:
                logger.error("Worker mode requires catalog_manager, run_id, and chunk_id")
                return False

            filename = f"{master_type}_{safe_sector}.parquet"
            # keyはダミーだが、save_delta側でpathsキーチェックに使われる可能性があるため
            # 存在しないキーだとエラーになるかも?
            # catalog_manager.save_delta の実装を見ると:
            # if custom_filename: filename = custom_filename
            # else: filename = paths[key]...
            # なので、custom_filenameがあれば paths[key] はアクセスされない。
            # ただし、念のため "master" を渡しておく。

            return catalog_manager.save_delta(
                key="master", df=new_data, run_id=run_id, chunk_id=chunk_id, custom_filename=filename
            )

        repo_path = f"master/{master_type}/sector={safe_sector}/data.parquet"

        # 1. 既存データのロード
        try:
            m_path = hf_hub_download(repo_id=self.hf_repo, filename=repo_path, repo_type="dataset", token=self.hf_token)
            master_df = pd.read_parquet(m_path)
            logger.debug(f"既存Master読み込み: {safe_sector} ({len(master_df)} rows)")
            combined_df = pd.concat([master_df, new_data], ignore_index=True)
        except Exception:
            logger.info(f"新規Master作成: {safe_sector} ({master_type})")
            combined_df = new_data

        # 2. 重複排除 (最新優先)
        subset = ["docid", "key", "context_ref"] if master_type == "financial_values" else ["docid", "key"]

        if "submitDateTime" in combined_df.columns:
            combined_df = combined_df.sort_values("submitDateTime", ascending=False)

        combined_df = combined_df.drop_duplicates(subset=subset, keep="first")

        # 3. 保存とアップロード
        local_file = self.data_path / f"master_{safe_sector}_{master_type}.parquet"

        for col in combined_df.columns:
            if combined_df[col].dtype == "object":
                combined_df[col] = combined_df[col].astype(str)

        combined_df.to_parquet(local_file, compression="zstd", index=False)

        if self.api:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.api.upload_file(
                        path_or_fileobj=str(local_file),
                        path_in_repo=repo_path,
                        repo_id=self.hf_repo,
                        repo_type="dataset",
                        token=self.hf_token,
                    )
                    logger.success(f"Master更新成功: {safe_sector} ({master_type})")
                    return True
                except Exception as e:
                    if isinstance(e, HfHubHTTPError) and e.response.status_code == 429:
                        wait_time = int(e.response.headers.get("Retry-After", 60)) + 5
                        logger.warning(
                            f"Master Rate limit exceeded. Waiting {wait_time}s... ({attempt + 1}/{max_retries})"
                        )
                        time.sleep(wait_time)
                        continue

                    logger.error(f"Masterアップロード失敗: {safe_sector} - {e}")
                    return False
            return False
        return True
