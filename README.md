# llm-finetuning

## 目的

Colab 上で SFT + QLoRA を試すための最小構成。

## 学習フロー

1. raw データを Drive に置く
2. preprocess して processed/train.jsonl を作る
3. run_train.py で学習
4. adapter を Drive に保存
5. run_eval.py で動作確認

## 保存先

- data: /content/drive/MyDrive/llm_ft_workspace/data
- outputs: /content/drive/MyDrive/llm_ft_workspace/outputs
