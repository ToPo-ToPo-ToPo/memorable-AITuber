
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

# ==========================================
# 1. 設定エリア
# ==========================================

# ベースモデル（Phase 1と同じRinna 3.6B）
model_name = "ToPo-ToPo/rinna-japanese-gpt-neox-3.6b"

# あなたが作成した学習データセットのパス
train_file_path = "outputs/character_data_mix.jsonl"

# 保存先（Phase 2用の名前）
output_dir = "models/lora-rinna-3.6b"
peft_name = "models/lora-rinna-3.6b-final"

# ==========================================
# 2. トークナイザーの準備
# ==========================================
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token # パディングトークンの設定

# コンテキスト長（長すぎる会話はカット）
CUTOFF_LEN = 512 

# ==========================================
# 3. プロンプト生成関数の定義
# ==========================================
def generate_prompt(data_point):
    """
    JSONLの messages 形式を、モデルに入力するテキスト形式に変換する関数
    Rinnaモデルは改行を <NL> として学習しているため、それに合わせます。
    """
    messages = data_point["messages"]
    
    # roleごとの内容を取得
    system_text = ""
    user_text = ""
    assistant_text = ""
    
    for msg in messages:
        if msg["role"] == "system":
            system_text = msg["content"]
        elif msg["role"] == "user":
            user_text = msg["content"]
        elif msg["role"] == "assistant":
            assistant_text = msg["content"]

    # プロンプトの組み立て
    # 形式: System: {設定} <NL> User: {質問} <NL> Assistant: {回答}
    full_text = (
        f"System: {system_text}<NL>"
        f"User: {user_text}<NL>"
        f"Assistant: {assistant_text}"
    )
    
    return full_text

# トークナイズ関数
def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
    )
    # label（正解データ）を作成（入力と同じものを設定し、後でCollatorで処理）
    result["labels"] = result["input_ids"].copy()
    return result

# ==========================================
# 4. データセットの読み込みと加工
# ==========================================

# ローカルのJSONLファイルを読み込む
data = load_dataset("json", data_files=train_file_path)

# データの確認（デバッグ用）
print("--- 元データ例 ---")
print(data["train"][0])
print("--- プロンプト変換後 ---")
print(generate_prompt(data["train"][0]))

# 学習データと検証データに分割（400件しかないので検証データは少なめに）
VAL_SET_SIZE = 20 # 20件だけ検証用に回す
train_val = data["train"].train_test_split(test_size=VAL_SET_SIZE, shuffle=True, seed=42)

# トークナイズ処理を適用
train_data = train_val["train"].shuffle().map(lambda x: tokenize(generate_prompt(x)))
val_data = train_val["test"].shuffle().map(lambda x: tokenize(generate_prompt(x)))

# ==========================================
# 5. モデルとLoRAの設定
# ==========================================

# モデルの読み込み
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16, # メモリ節約のためfloat16
)

# LoRAの設定（ここがPhase 2のキモ！）
lora_config = LoraConfig(
    r=32,                  # Rank: キャラクター性を強く出すため 8 -> 32 に増強
    lora_alpha=64,         # Alpha: 重みを強くするため 16 -> 64 に増強
    target_modules=[       # 学習させる層を増やす
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
model.print_trainable_parameters() # 学習パラメータ数の表示

# ==========================================
# 6. トレーナーの設定と実行
# ==========================================

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4, # GPUメモリに応じて調整（きついなら2へ）
    gradient_accumulation_steps=4,
    num_train_epochs=10,           # データが少ないのでEpoch数を増やす（3 -> 10）
    learning_rate=3e-4,            # 学習率
    fp16=True,                     # GPU高速化
    logging_steps=10,
    save_strategy="epoch",         # エポックごとに保存
    evaluation_strategy="epoch",   # エポックごとに評価
    save_total_limit=2,            # 最新の2つだけ残す
    report_to="none",
    remove_unused_columns=False    # mapで作成したカラムを消さないように
)

trainer = Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# 学習開始
model.config.use_cache = False
trainer.train()
model.config.use_cache = True

# ==========================================
# 7. モデルの保存
# ==========================================
trainer.model.save_pretrained(peft_name)
tokenizer.save_pretrained(peft_name)

print(f"学習完了！モデルは {peft_name} に保存されました。")