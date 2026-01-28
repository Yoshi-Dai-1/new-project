from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from loguru import logger


class HistoryEngine:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.jpx_url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
        self.nikkei_url = "https://indexes.nikkei.co.jp/nkave/statistics/datalist/constituent?list=225&type=csv"
        logger.info("HistoryEngine を初期化しました。")

    def fetch_jpx_master(self) -> pd.DataFrame:
        """JPXから最新の銘柄一覧を取得"""
        logger.info("JPX銘柄マスタを取得中...")
        r = requests.get(self.jpx_url, stream=True, verify=False)
        self.data_path.mkdir(parents=True, exist_ok=True)
        xls_path = self.data_path / "jpx_master.xls"
        with xls_path.open("wb") as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)

        df = pd.read_excel(xls_path, dtype={"コード": str})
        df = df.rename(
            columns={"コード": "code", "銘柄名": "company_name", "33業種区分": "sector", "市場・商品区分": "market"}
        )
        df["code"] = df["code"].str[:4]
        logger.success(f"JPXマスタ取得完了: {len(df)} 銘柄")
        return df[["code", "company_name", "sector", "market"]]

    def generate_listing_events(self, old_master: pd.DataFrame, new_master: pd.DataFrame) -> pd.DataFrame:
        """マスタの差分から上場・廃止イベントを生成"""
        if old_master.empty:
            return pd.DataFrame()

        old_codes = set(old_master["code"])
        new_codes = set(new_master["code"])

        events = []
        today = datetime.now().strftime("%Y-%m-%d")

        # 新規上場
        for code in new_codes - old_codes:
            events.append({"code": code, "type": "LISTING", "event_date": today, "note": "Newly Listed"})

        # 廃止
        for code in old_codes - new_codes:
            events.append({"code": code, "type": "DELISTING", "event_date": today, "note": "Delisted / Merged"})

        return pd.DataFrame(events)

    def fetch_nikkei_225_events(self) -> pd.DataFrame:
        """日経225構成銘柄を取得（簡易版）"""
        # 実際にはCSVのパーシングが必要だが、ここでは空リストまたは簡易取得を想定
        return pd.DataFrame(columns=["code"])

    def generate_index_events(self, index_name: str, old_list: pd.DataFrame, new_list: pd.DataFrame) -> pd.DataFrame:
        """指数採用・除外イベント生成"""
        old_set = set(old_list["code"]) if not old_list.empty else set()
        new_set = set(new_list["code"]) if not new_list.empty else set()

        events = []
        today = datetime.now().strftime("%Y-%m-%d")

        for code in new_set - old_set:
            events.append({"index_name": index_name, "code": code, "type": "ADD", "event_date": today})
        for code in old_set - new_set:
            events.append({"index_name": index_name, "code": code, "type": "REMOVE", "event_date": today})

        return pd.DataFrame(events)
