from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from loguru import logger

from models import CatalogRecord, StockMasterRecord


class CatalogManager:
    def __init__(self, hf_repo: str, hf_token: str, data_path: Path):
        self.hf_repo = hf_repo
        self.hf_token = hf_token
        self.data_path = data_path
        self.api = HfApi() if hf_repo and hf_token else None

        # ファイルパス定義
        self.paths = {
            "catalog": "catalog/documents_index.parquet",
            "master": "meta/stocks_master.parquet",
            "listing": "meta/listing_history.parquet",
            "index": "meta/index_history.parquet",
            "name": "meta/name_history.parquet",
        }

        self.catalog_df = self._load_parquet("catalog")
        self.master_df = self._load_parquet("master")
        logger.info("CatalogManager を初期化しました。")

    def _load_parquet(self, key: str) -> pd.DataFrame:
        filename = self.paths[key]
        try:
            local_path = hf_hub_download(
                repo_id=self.hf_repo, filename=filename, repo_type="dataset", token=self.hf_token
            )
            df = pd.read_parquet(local_path)
            logger.debug(f"ロード成功: {filename} ({len(df)} rows)")
            return df
        except Exception:
            logger.info(f"既存ファイルが見つかりません。新規作成します: {filename}")
            if key == "catalog":
                cols = list(CatalogRecord.model_fields.keys())
                return pd.DataFrame(columns=cols)
            elif key == "master":
                cols = list(StockMasterRecord.model_fields.keys())
                return pd.DataFrame(columns=cols)
            elif key == "listing":
                return pd.DataFrame(columns=["code", "type", "event_date"])
            elif key == "index":
                return pd.DataFrame(columns=["index_name", "code", "type", "event_date"])
            elif key == "name":
                return pd.DataFrame(columns=["code", "old_name", "new_name", "change_date"])
            return pd.DataFrame()

    def is_processed(self, doc_id: str) -> bool:
        if self.catalog_df.empty:
            return False
        return doc_id in self.catalog_df["doc_id"].values

    def update_catalog(self, new_records: List[Dict]) -> bool:
        """カタログを更新 (Pydanticバリデーション実施)"""
        if not new_records:
            return True

        validated = []
        for rec in new_records:
            try:
                validated.append(CatalogRecord(**rec).model_dump())
            except Exception as e:
                logger.error(f"カタログレコードのバリデーション失敗 (doc_id: {rec.get('doc_id')}): {e}")

        if not validated:
            return False

        new_df = pd.DataFrame(validated)
        self.catalog_df = pd.concat([self.catalog_df, new_df], ignore_index=True).drop_duplicates(
            subset=["doc_id"], keep="last"
        )
        return self._save_and_upload("catalog", self.catalog_df)

    def _save_and_upload(self, key: str, df: pd.DataFrame) -> bool:
        filename = self.paths[key]
        local_file = self.data_path / Path(filename).name

        # 型の安定化
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].astype(str)

        df.to_parquet(local_file, index=False, compression="zstd")

        if self.api:
            try:
                self.api.upload_file(
                    path_or_fileobj=str(local_file),
                    path_in_repo=filename,
                    repo_id=self.hf_repo,
                    repo_type="dataset",
                    token=self.hf_token,
                )
                logger.success(f"アップロード成功: {filename}")
                return True
            except Exception as e:
                logger.error(f"アップロード失敗: {filename} - {e}")
                return False
        return True

    def upload_raw(self, local_path: Path, repo_path: str) -> bool:
        """ローカルの生データを Hugging Face の raw/ フォルダにアップロード"""
        if not local_path.exists():
            logger.error(f"ファイルが存在しないためアップロードできません: {local_path}")
            return False

        if self.api:
            try:
                self.api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=repo_path,
                    repo_id=self.hf_repo,
                    repo_type="dataset",
                    token=self.hf_token,
                )
                logger.debug(f"RAWアップロード成功: {repo_path}")
                return True
            except Exception as e:
                logger.error(f"RAWアップロード失敗: {repo_path} - {e}")
                return False
        return True

    def update_listing_history(self, new_events: pd.DataFrame) -> bool:
        if new_events.empty:
            return True
        history = self._load_parquet("listing")
        history = pd.concat([history, new_events], ignore_index=True).drop_duplicates()
        return self._save_and_upload("listing", history)

    def update_index_history(self, new_events: pd.DataFrame) -> bool:
        if new_events.empty:
            return True
        history = self._load_parquet("index")
        history = pd.concat([history, new_events], ignore_index=True).drop_duplicates()
        return self._save_and_upload("index", history)

    def get_listing_history(self) -> pd.DataFrame:
        """現在の上場履歴マスタを取得"""
        return self._load_parquet("listing")

    def get_index_history(self) -> pd.DataFrame:
        """現在の指数採用履歴マスタを取得"""
        return self._load_parquet("index")

    def update_stocks_master(self, new_master: pd.DataFrame):
        """マスタ更新 (Pydantic バリデーション実施)"""
        if new_master.empty:
            return

        records = new_master.to_dict("records")
        validated = []
        for rec in records:
            try:
                validated.append(StockMasterRecord(**rec).model_dump())
            except Exception as e:
                logger.error(f"銘柄マスタのバリデーション失敗 (code: {rec.get('code')}): {e}")

        if not validated:
            return
        valid_df = pd.DataFrame(validated)

        # 社名変更チェック
        if not self.master_df.empty:
            merged = pd.merge(
                self.master_df[["code", "company_name"]],
                valid_df[["code", "company_name"]],
                on="code",
                suffixes=("_old", "_new"),
            )
            changed = merged[merged["company_name_old"] != merged["company_name_new"]]
            if not changed.empty:
                today = datetime.now().strftime("%Y-%m-%d")
                name_history = self._load_parquet("name")
                for _, row in changed.iterrows():
                    name_history = pd.concat(
                        [
                            name_history,
                            pd.DataFrame(
                                [
                                    {
                                        "code": row["code"],
                                        "old_name": row["company_name_old"],
                                        "new_name": row["company_name_new"],
                                        "change_date": today,
                                    }
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )
                self._save_and_upload("name", name_history.drop_duplicates())

        self.master_df = valid_df
        self._save_and_upload("master", self.master_df)

    def get_last_index_list(self, index_name: str) -> pd.DataFrame:
        """指定指数の構成銘柄を取得 (Phase 3用)"""
        return pd.DataFrame(columns=["code"])

    def get_sector(self, code: str) -> str:
        """証券コードから業種取得"""
        if self.master_df.empty:
            return "その他"
        row = self.master_df[self.master_df["code"] == code]
        if not row.empty:
            return str(row.iloc[0]["sector"])
        return "その他"
