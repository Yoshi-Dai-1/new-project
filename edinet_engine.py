from pathlib import Path
from typing import Dict, List

import requests
from loguru import logger

# 内部モジュール（外部ライブラリ）のインポート
from edinet_xbrl_prep.edinet_xbrl_prep.edinet_api import edinet_response_metadata, request_term
from edinet_xbrl_prep.edinet_xbrl_prep.link_base_file_analyzer import account_list_common
from models import EdinetDocument


class EdinetEngine:
    def __init__(self, api_key: str, data_path: Path, taxonomy_urls: Dict[str, str] = None):
        self.api_key = api_key
        self.data_path = data_path
        self.taxonomy_urls = taxonomy_urls or {}
        self._apply_monkypatches()
        logger.info("EdinetEngine を初期化しました。")

    def _apply_monkypatches(self):
        """ライブラリのバグや制約を修正するためのパッチ適用"""

        # サブモジュール内の verify=False を物理的に無効化する
        import edinet_xbrl_prep.edinet_xbrl_prep.edinet_api as edinet_api_mod

        # get_edinet_metadata と request_doc の内部で直接 requests.Session().get している箇所をラップ
        original_session = requests.Session

        class SecureSession(original_session):
            def get(self, url, **kwargs):
                # EDINET関連のURLなら verify=True を強制
                if "api.edinet-fsa.go.jp" in url:
                    kwargs["verify"] = True
                return super().get(url, **kwargs)

        # 影響範囲を限定するため、モジュール内の requests.Session を差し替える
        edinet_api_mod.requests.Session = SecureSession

    def fetch_metadata(self, start_date: str, end_date: str) -> List[Dict]:
        """指定期間の全書類メタデータを取得し、Pydanticでバリデーション"""
        logger.info(f"EDINETメタデータ取得開始: {start_date} ~ {end_date}")
        res_results = request_term(api_key=self.api_key, start_date_str=start_date, end_date_str=end_date)

        tse_url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
        meta = edinet_response_metadata(tse_sector_url=tse_url, tmp_path_str=str(self.data_path))
        meta.set_data(res_results)

        df = meta.get_metadata_pandas_df()
        if df.empty:
            logger.warning("対象期間の書類は見つかりませんでした。")
            return []

        records = df.to_dict("records")
        validated_records = []
        for rec in records:
            try:
                # Pydantic モデルでバリデーション & 正規化
                doc = EdinetDocument(**rec)
                validated_records.append(doc.model_dump(by_alias=True))
            except Exception as e:
                logger.error(f"書類メタデータのバリデーション失敗 (docID: {rec.get('docID')}): {e}")

        logger.success(f"メタデータ取得完了: {len(validated_records)} 件")
        return validated_records

    def download_doc(self, doc_id: str, save_path: Path, doc_type: int = 1) -> bool:
        """書類をダウンロード保存 (1=XBRL, 2=PDF)"""
        url = f"https://api.edinet-fsa.go.jp/api/v2/documents/{doc_id}"
        params = {"type": doc_type, "Subscription-Key": self.api_key}

        try:
            r = requests.get(url, params=params, timeout=(20, 90), stream=True)
            if r.status_code == 200:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 64):
                        f.write(chunk)
                logger.debug(f"取得成功: {doc_id} (type={doc_type})")
                return True
            else:
                logger.error(f"DL失敗: {doc_id} (HTTP {r.status_code})")
                return False
        except Exception:
            logger.exception(f"DLエラー: {doc_id}")
            return False

    def get_account_list(self, taxonomy_year: str):
        """解析用タクソノミの取得"""
        try:
            # ライブラリ内部に直接 Path を渡す（ユーザーの指摘通り、初期状態で動作していたはずの実装に戻す）
            acc = account_list_common(str(self.data_path), taxonomy_year)
            return acc
        except Exception:
            logger.exception(f"タクソノミ取得エラー (Year: {taxonomy_year})")
            return None
