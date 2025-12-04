
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# ==========================================
# 1. 設定エリア
# ==========================================
model_name = "ToPo-ToPo/rinna-japanese-gpt-neox-3.6b-lora-sft-v1"

# ★ここにあなたのHugging FaceのリポジトリIDを入力してください
# 例: "user_name/my-aituber-dataset"
HF_DATASET_ID = "ToPo-ToPo/character-data-Nemu"

# ★リポジトリ内のファイル名 (jsonl)
# 例: "data.jsonl" や "train.jsonl"
HF_DATA_FILE = "data.jsonl"

# 保存先
peft_name = "models/lora-rinna-3.6b-phase2-neum"
output_dir = "models/lora-rinna-3.6b-phase2-results"

# コンテキスト長 (Systemプロンプトが入るため512必須)
CUTOFF_LEN = 1024

# ==========================================
# 2. モデルとトークナイザーの準備
# ==========================================
print("モデルを読み込んでいます...")

# Mac (MPS) 用の設定: float32で安定化
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="mps",         
    torch_dtype=torch.float32 
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==========================================
# 3. データセット読み込みとプロンプト変換
# ==========================================
print(f"Hugging Face ({HF_DATASET_ID}) からデータを取得しています...")

# Hugging FaceからJSONLを読み込む
dataset = load_dataset(
    HF_DATASET_ID,
    data_files=HF_DATA_FILE,
    split="train" # 最初からtrain分割として読み込む
)

def generate_and_tokenize_prompt(data_point):
    """
    JSONL(messages) -> Phase 1形式(テキスト) -> Tokenize(512固定)
    """
    messages = data_point["messages"]
    
    system_text = ""
    user_text = ""
    assistant_text = ""

    # roleごとに中身を抽出
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            system_text = content
        elif role == "user":
            user_text = content
        elif role == "assistant":
            assistant_text = content

    # Phase 1 のフォーマットに変換
    full_prompt = (
        f"### 指示:\n"
        f"{system_text}\n\n"
        f"### 入力:\n"
        f"{user_text}\n\n"
        f"### 回答:\n"
        f"{assistant_text}"
    )
    
    # 改行コード変換
    full_prompt = full_prompt.replace('\n', '<NL>')
    full_prompt = full_prompt + tokenizer.eos_token

    # ---------------------------------------------------------
    # ★重要: padding="max_length" で長さを強制的に512に揃える
    # これで ValueError (Shape mismatch) を回避
    # ---------------------------------------------------------
    tokenized_full_prompt = tokenizer(
        full_prompt,
        truncation=True,
        max_length=CUTOFF_LEN, # 512
        padding="max_length",  # 空白埋め有効
    )
    
    # ---------------------------------------------------------
    # ★重要: パディング部分を学習対象外(-100)にする
    # ---------------------------------------------------------
    input_ids = tokenized_full_prompt["input_ids"]
    labels = list(input_ids)
    pad_token_id = tokenizer.pad_token_id
    
    labels = [
        -100 if token == pad_token_id else token 
        for token in labels
    ]
    tokenized_full_prompt["labels"] = labels
    
    return tokenized_full_prompt

# データセットの分割
print("データのトークナイズ処理を実行中...")
train_val = dataset.train_test_split(test_size=20, shuffle=True, seed=42)

# マップ処理実行
train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
eval_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)

print(f"学習データ数: {len(train_data)}")
print(f"検証データ数: {len(eval_data)}")

# ==========================================
# 4. LoRA設定 (Phase 2 強力版)
# ==========================================
print("LoRAを設定しています...")
lora_config = LoraConfig(
    r=32,                   # キャラクター性を強く
    lora_alpha=64,          # 重みを強く
    target_modules=[        # 全層学習
        "query_key_value", 
        "dense", 
        "dense_h_to_4h", 
        "dense_4h_to_h"
    ], 
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==========================================
# 5. Trainer設定と実行
# ==========================================
print("学習を開始します...")

trainer = Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    args=TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,       # 10周
        learning_rate=3e-4,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        
        # Mac (MPS) 用設定
        optim="adamw_torch",         
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=4,
        fp16=False, # MPS安定のためFalse
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

# ==========================================
# 6. 保存
# ==========================================
print("モデルを保存しています...")
trainer.model.save_pretrained(peft_name)
tokenizer.save_pretrained(peft_name)

print(f"完了しました！ 保存先: {peft_name}")