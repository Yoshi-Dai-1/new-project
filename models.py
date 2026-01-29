from typing import Optional

from pydantic import BaseModel


class EdinetDocument(BaseModel):
    """EDINET APIから取得される書類メタデータのバリデーションモデル"""

    docID: str
    docDescription: Optional[str] = None
    filerName: Optional[str] = None
    submitDateTime: str
    secCode: Optional[str] = None
    edinetCode: Optional[str] = None
    docTypeCode: Optional[str] = None
    ordinanceCode: Optional[str] = None
    formCode: Optional[str] = None
    xbrlFlag: Optional[str] = "0"
    pdfFlag: Optional[str] = "0"


class CatalogRecord(BaseModel):
    """統合ドキュメントカタログ (documents_index.parquet) のレコードモデル"""

    doc_id: str
    source: str
    code: str
    edinet_code: Optional[str] = None
    company_name: str
    doc_type: str
    title: str
    submit_at: str
    raw_zip_path: Optional[str] = None
    pdf_path: Optional[str] = None
    processed_status: Optional[str] = "success"


class StockMasterRecord(BaseModel):
    """銘柄マスタ (stocks_master.parquet) のレコードモデル"""

    code: str
    company_name: str
    sector: Optional[str] = "その他"
    market: Optional[str] = None
    is_active: bool = True


class ListingEvent(BaseModel):
    """上場・廃止イベントの記録モデル"""

    code: str
    type: str  # LISTING, DELISTING
    event_date: str
    note: Optional[str] = None


class IndexEvent(BaseModel):
    """指数採用・除外イベントの記録モデル"""

    index_name: str
    code: str
    type: str  # ADD, REMOVE
    event_date: str
