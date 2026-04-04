# llm-finetuning

## 目的

Colab 上で SFT + QLoRA を試すための最小構成。

## セットアップ

### ローカル

```bash
uv sync --dev
```

notebook を開く場合:

```bash
uv run jupyter lab
```

スクリプト実行例:

```bash
uv run python scripts/run_preprocess.py
uv run python scripts/run_train.py
uv run python scripts/run_eval.py
```

### Colab

`notebooks/train_colab.ipynb` は `uv` をインストールして `uv sync --dev` する前提。

## 学習フロー

1. raw データを Drive に置く
2. preprocess して processed/train.jsonl を作る
3. run_train.py で学習
4. adapter を Drive に保存
5. run_eval.py で動作確認

## 保存先

- data: /content/drive/MyDrive/llm_ft_workspace/data
- outputs: /content/drive/MyDrive/llm_ft_workspace/outputs
