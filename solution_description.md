# v21 解法説明

## 概要

マウスの脳活動データ（44脳領域 × 左右 = 88チャンネル × 10時点）からレバー押し量を予測する回帰タスク。
2段Stackingとアンサンブルにより、Public LB: 0.705を達成。

## モデル構成

### Stage 1: 3モデル並列学習（5-Fold GroupKFold）

| モデル | 種類 | 役割 |
|--------|------|------|
| BiGRU (64d) | 双方向GRU | 10時点の時系列パターンを前後方向から捉える |
| MLP | 全結合NN | 各時点の特徴量の組み合わせで瞬間的に予測 |
| LightGBM | 勾配ブースティング | 200以上の手作り特徴量から非線形な交互作用を捉える |

### Stage 2: Ridge回帰によるStacking

Stage 1の3モデルのOOF（Out-of-Fold）予測を入力として、Ridge回帰で最適に統合。
結果としてBiGRUの重みが最大となった。

### 追加MLP群（4種類）

| モデル | 特徴 |
|--------|------|
| v17a MLP | v10bとは異なるアーキテクチャ |
| Step3 MLP | 3段階stacking構造 |
| Stat MLP | 統計特徴量を重視 |
| Huber MLP | Huber lossで学習（外れ値に頑健） |

### 最終ブレンド（Optuna TPE, 3000 trials）

| モデル | 重み |
|--------|------|
| v10b stacking (seed=42) | 71.4% |
| Step3 MLP | 12.5% |
| v17a MLP | 7.1% |
| Stat MLP | 4.5% |
| Huber MLP | 4.5% |

## 前処理

- マウスごとのz-score正規化（個体差の吸収）
- マウス×日ごとの正規化（日内変動の吸収）
- マウス別Rest値のヒストグラムピーク検出

## 特徴量（LightGBM用、200以上）

- 時系列特徴: velocity, acceleration, rolling std, lag特徴量
- 脳領域の相互作用: 左右の和・比率・積、運動野×体性感覚野の比率
- マウス統計量: 活動率、活動時の平均・標準偏差・中央値、レバー最大値

## 後処理

- Power Transform: `sign(x) * |x|^1.08`
- Scale: `pred * 0.995`
- Clip: `[0.0, 4.5]`

## CV戦略

- 5-Fold GroupKFold（sample_id単位でグルーピング）
- OOF予測で全データのCV MSEを算出

## 使用技術

- Python / PyTorch / LightGBM / scikit-learn / Optuna
- 実行環境: Google Colab (T4 GPU)
- 全モデルをコンペ提供データのみでゼロから学習（外部データ・事前学習済みモデル不使用）