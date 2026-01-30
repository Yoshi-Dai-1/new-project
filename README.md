# ARIA - Analytics & Research Intelligence Assistant

**ARIA** (アリア) は、日本の金融市場データを自動収集・分析するための完全無料のデータレイクハウスプラットフォームです。

## プロジェクト名の由来

**ARIA** は以下の頭文字から命名されました:
- **A**nalytics: データ分析
- **R**esearch: 投資調査
- **I**ntelligence: 知的情報処理
- **A**ssistant: 投資判断の支援

また、音楽用語の「アリア（詠唱）」にも由来しており、複雑な金融データを美しく調和のとれた洞察へと変換する、という理念を表現しています。

## 特徴

- **完全自動化**: GitHub Actionsによる毎日の自動データ収集
- **完全無料**: オープンソース、永続的に無料で利用可能
- **データレイクハウス**: RAW/Master/Metaの3層アーキテクチャ
- **Hugging Face統合**: データセット `Yoshi-Dai/financial-lakehouse` で公開
- **投資判断支援**: 財務諸表、定性情報、市場履歴を統合分析

## データソース

- **EDINET**: 金融庁が運営する有価証券報告書等の開示システム
- **JPX**: 日本取引所グループの銘柄マスタ
- **Nikkei 225**: 日経平均株価の構成銘柄履歴

## データ構造

```
financial-lakehouse/
├── raw/                    # 生データ（ZIP, PDF）
│   └── edinet/
│       └── YYYY/MM/
├── catalog/                # ドキュメントインデックス
│   └── documents_index.parquet
├── meta/                   # メタデータ
│   ├── stocks_master.parquet
│   ├── listing_history.parquet
│   └── index_history.parquet
└── master/                 # 分析用マスタデータ
    ├── financial_values/   # 財務数値（BS, PL, CF, SS）
    │   └── sector=XXX/
    └── qualitative_text/   # 定性情報（注記）
        └── sector=XXX/
```

## 使い方

### 環境変数の設定

```bash
HF_REPO=Yoshi-Dai/financial-lakehouse
HF_TOKEN=your_huggingface_token
```

### データ収集の実行

```bash
# 特定期間のデータを収集
python main.py --start 2024-01-01 --end 2024-01-31

# 特定の書類IDを指定
python main.py --id-list S100XXXX,S100YYYY
```

## ローカルフォルダ名の変更手順

プロジェクトフォルダ名を `new-project` から `aria` に変更する場合:

### 1. フォルダ名の変更

```powershell
# PowerShellで実行
cd C:\projects
Rename-Item -Path "new-project" -NewName "aria"
```

### 2. 必要な手続き

フォルダ名変更後、以下の手続きは**不要**です:
- ✅ Gitリポジトリは自動的に新しいパスで動作します
- ✅ Python仮想環境（venv）も自動的に新しいパスで動作します
- ✅ 環境変数（HF_REPO, HF_TOKEN）はフォルダ名に依存しません

ただし、以下の点を確認してください:
- VSCodeやIDEで開いている場合は、新しいパスで再度開き直してください
- ターミナルで作業中の場合は、新しいパスに移動してください: `cd C:\projects\aria`

### 3. 動作確認

```powershell
cd C:\projects\aria
python main.py --list-only --start 2024-01-01 --end 2024-01-01
```

## ライセンス

MIT License

## 貢献

プルリクエストを歓迎します。大きな変更の場合は、まずissueを開いて変更内容を議論してください。

## 免責事項

このプロジェクトは投資判断の参考情報を提供するものであり、投資助言ではありません。投資は自己責任で行ってください。
