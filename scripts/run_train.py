from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
import torch
from datasets import Dataset
from pathlib import Path
import pandas as pd

model_name = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

model.config.pad_token_id = tokenizer.pad_token_id

model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    modules_to_save=["score"],
)

model = get_peft_model(model, peft_config)


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = REPO_ROOT / "data" / "processed" / "preprcessed_train.parquet"

df = pd.read_parquet(INPUT_PATH)

train_dataset = Dataset.from_pandas(df[["text", "label"]])


def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=2048,
        padding=False,
    )


tokenized_train = train_dataset.map(tokenize_fn, batched=True)

training_args = TrainingArguments(
    output_dir="./outputs/qwen25_7b_cls",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=200,
    bf16=True,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    tokenizer=tokenizer,
)

trainer.train()
