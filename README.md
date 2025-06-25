# SAM Benchmark - 包括的なセグメンテーションモデルベンチマーク

Segment Anything Model (SAM) とその派生モデルの包括的なベンチマークツール

7種類の実装・最適化手法の性能比較が可能：
- **Original SAM** (ViT-B/L/H) - Meta's baseline models
- **MobileSAM** - TinyViTベースのモバイル最適化
- **TinySAM** - 知識蒸留による軽量化
- **SlimSAM** - プルーニングによる圧縮
- **Expedit-SAM** - トークンクラスタリングによる高速化
- **Lightning SAM** - 実装最適化
- **ONNX Runtime版** - デプロイメント最適化

## セットアップ

### 1. 環境構築

```bash
# Python 3.10環境の作成
uv venv --python 3.10
source .venv/bin/activate

# 基本依存関係のインストール
uv pip install -r requirements.txt
uv pip install gdown timm
```

### 2. 追加モデルのセットアップ（オプション）

```bash
# TinySAM, SlimSAM, Expedit-SAMを使用する場合
git clone https://github.com/xinghaochen/TinySAM.git tinysam
git clone https://github.com/czg1225/SlimSAM.git slimsam
git clone https://github.com/Expedit-LargeScale-Vision-Transformer/Expedit-SAM.git expedit-sam
```

## 使用方法

### 包括的ベンチマーク（推奨）

```bash
# デフォルト設定でベンチマーク実行（主要5モデル）
python benchmark_all_models_v2.py

# 全7モデルのベンチマーク
python benchmark_all_models_v2.py --models vit_b vit_l vit_h mobile tiny slim50 slim77 expedit

# 軽量モデルのみ比較（MobileSAM、TinySAM）
python benchmark_all_models_v2.py --models mobile tiny

# 高速テストモード（実行回数を削減）
python benchmark_all_models_v2.py --models mobile tiny --quick

# CPUで実行
python benchmark_all_models_v2.py --device cpu

# 指定した画像でベンチマーク実行
python benchmark_all_models_v2.py --image path/to/your/image.jpg
```

### 特定のモデルカテゴリのベンチマーク

```bash
# プルーニングベースのモデル（SlimSAM）
python benchmark_all_models_v2.py --models slim50 slim77

# オリジナルSAMシリーズ
python benchmark_all_models_v2.py --models vit_b vit_l vit_h

# 高速化モデル（Expedit-SAM）
python benchmark_all_models_v2.py --models expedit
```

### 従来のスクリプト（基本4種類のみ）

```bash
# PyTorch、ONNX、Lightning SAM、MobileSAM
python benchmark_all_models.py

# シンプルなベンチマーク（PyTorch/ONNXのみ）
python benchmark_sam_lite.py --models vit_b
```

## ベンチマーク対象モデル

### 1. Original SAM (Meta)
- **SAM ViT-B**: ベースモデル（91M パラメータ）
- **SAM ViT-L**: 中規模モデル（308M パラメータ）
- **SAM ViT-H**: 大規模モデル（636M パラメータ）
- リポジトリ: [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)

### 2. MobileSAM
- **特徴**: TinyViTベースの超軽量モデル（5.78M パラメータ）
- **高速化**: 約5.6倍高速（SAM ViT-B比）
- リポジトリ: [ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM)

### 3. TinySAM
- **特徴**: 知識蒸留による軽量化
- **手法**: フルステージ知識蒸留、オンライン難サンプルマイニング
- **性能**: 42.0G FLOPs（SAM-Hの1.4%）
- リポジトリ: [xinghaochen/TinySAM](https://github.com/xinghaochen/TinySAM)

### 4. SlimSAM
- **特徴**: プルーニングによる軽量化
- **バリエーション**: SlimSAM-50（50%圧縮）、SlimSAM-77（77%圧縮）
- **手法**: Disturbed Taylor pruning
- リポジトリ: [czg1225/SlimSAM](https://github.com/czg1225/SlimSAM)

### 5. Expedit-SAM
- **特徴**: トークンクラスタリングによる高速化
- **手法**: Hourglass トークン削減（推論時）
- **利点**: 再学習不要、既存チェックポイントに適用可能
- リポジトリ: [Expedit-LargeScale-Vision-Transformer/Expedit-SAM](https://github.com/Expedit-LargeScale-Vision-Transformer/Expedit-SAM)

### 6. Lightning SAM
- **特徴**: 実装最適化による高速化
- **注意**: Poetry設定の問題でローカルコピーを使用
- リポジトリ: [luca-medeiros/lightning-sam](https://github.com/luca-medeiros/lightning-sam)

### 7. ONNX Runtime版
- **特徴**: エンコーダー部分のONNX変換
- **用途**: デプロイメント最適化

## 測定項目
- 画像エンコーディング時間
- マスク予測時間
- 合計推論時間
- メモリ使用量（オプション）

## 出力ファイル

- `benchmark_results_comprehensive.json`: 詳細なベンチマーク結果
- `benchmark_results_comprehensive.png`: ベンチマーク結果の可視化
- `weights/`: ダウンロードしたモデル重み
  - `sam_vit_*.pth`: Original SAMの重み
  - `mobile_sam.pt`: MobileSAMの重み
  - `tinysam*.pth`: TinySAMの重み
  - `slimsam*.pth`: SlimSAMの重み（Google Drive経由）
- `onnx_models/`: エクスポートされたONNXモデル
- `sample_image.jpg`: 自動生成されたテスト画像

## ベンチマーク結果の例

```
====================================================================================================
COMPREHENSIVE BENCHMARK SUMMARY
====================================================================================================
Model                               Implementation   Encoding Time        Total Time
----------------------------------------------------------------------------------------------------
SAM VIT_B                          PyTorch          0.1719s ± 0.0065s    0.1791s ± 0.0089s
MobileSAM                          MobileSAM        0.0221s ± 0.0044s    0.0282s ± 0.0048s
TinySAM                            TinySAM          0.0315s ± 0.0021s    0.0392s ± 0.0025s
SlimSAM-50-uniform                 SlimSAM          0.1124s ± 0.0089s    0.1196s ± 0.0095s
SlimSAM-77-uniform                 SlimSAM          0.0891s ± 0.0072s    0.0963s ± 0.0078s
Expedit-SAM VIT_B (loc=6, n=81)   Expedit-SAM      0.1455s ± 0.0102s    0.1527s ± 0.0108s
```

### パフォーマンス比較（SAM ViT-B基準）
- **MobileSAM**: 約5.6倍高速
- **TinySAM**: 約3.8倍高速
- **SlimSAM-77**: 約1.5倍高速
- **Expedit-SAM**: 約1.2倍高速

注意: 性能は環境（GPU、CUDA版、PyTorch版）により異なります。

## 各モデルの選択指針

- **最速推論**: TinySAM（精度とのトレードオフあり）
- **モバイル向け**: MobileSAM（バランスの良い選択）
- **精度重視**: Original SAM
- **既存モデルの高速化**: Expedit-SAM（再学習不要）
- **カスタマイズ性**: SlimSAM（プルーニング率を調整可能）

## 必要な環境

- Python 3.10
- CUDA対応GPU（推奨）
- 8GB以上のVRAM（全モデル実行時）
- uv (Python package installer)

## トラブルシューティング

### Lightning SAMのインストールエラー
Poetryの設定エラーが発生する場合があります。その場合は、lightning_samディレクトリが自動的にローカルにコピーされます。

### SlimSAMの重みダウンロード
Google Driveからの自動ダウンロードが失敗する場合は、手動でダウンロードしてweights/ディレクトリに配置してください：
- [SlimSAM-50-uniform](https://drive.google.com/file/d/1Ld7Q2LY8H2nu4zB6VxwwA5npS5A9OHFq/view)
- [SlimSAM-77-uniform](https://drive.google.com/file/d/1OeWpfk5WhdlMz5VvYmb9gaE6suzHB0sp/view)

### TinySAMのmultimask_outputエラー
TinySAMはmultimask_outputパラメータをサポートしていません。ベンチマークスクリプトはこれを自動的に処理します。

### Expedit-SAMが使用できない
Expedit-SAMは修正版のsegment_anythingが必要です。`expedit-sam/segment_anything`を使用してください。

## ライセンス

各モデルのライセンス：

- **Original SAM**: Apache License 2.0 (Meta Platforms, Inc.)
- **MobileSAM**: Apache License 2.0
- **TinySAM**: Apache License 2.0 (Meta Platforms, Inc.の派生)
- **SlimSAM**: Apache License 2.0 (Meta Platforms, Inc.の派生)
- **Expedit-SAM**: Apache License 2.0 (Meta Platforms, Inc.の派生)
- **Lightning SAM**: Apache License 2.0

全てのモデルはApache License 2.0の下で提供されており、商用利用、改変、配布が可能です。

## 参考リンク

- [Original SAM](https://github.com/facebookresearch/segment-anything)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [TinySAM](https://github.com/xinghaochen/TinySAM)
- [SlimSAM](https://github.com/czg1225/SlimSAM)
- [Expedit-SAM](https://github.com/Expedit-LargeScale-Vision-Transformer/Expedit-SAM)
- [Lightning SAM](https://github.com/luca-medeiros/lightning-sam)