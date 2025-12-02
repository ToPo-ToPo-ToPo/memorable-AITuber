import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, TaskType
from trl import SFTTrainer

#-------------------------------------------------------------------------------------
# 設定項目
#-------------------------------------------------------------------------------------
model_name = "ToPo-ToPo/rinna-japanese-gpt-neox-3.6b"
dataset_name = "kunishou/databricks-dolly-15k-ja"
peft_name = "lora-rinna-3.6b"
output_dir = "lora-rinna-3.6b-results"
CUTOFF_LEN = 512 

#---------------------------------------------------------
# モデルの準備 (Float32 / MPS)
#---------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="mps",         
    torch_dtype=torch.float32 
)
model.config.use_cache = False

#---------------------------------------------------------
# トークナイザーの準備
#---------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#---------------------------------------------------------
# データセットの準備とフォーマット関数 (ここを修正)
#---------------------------------------------------------
dataset = load_dataset(dataset_name)

# 修正箇所: 単一のデータを処理するように変更
def formatting_prompts_func(example):
    # example は辞書型 {'instruction': '...', 'input': '...', ...} です
    # リストではなく単一の文字列が入ってきます
    
    instruction = example['instruction']
    input_context = example['input']
    output = example['output']

    if input_context:
        prompt = (
            f"### 指示:\n{instruction}\n\n"
            f"### 入力:\n{input_context}\n\n"
            f"### 回答:\n{output}"
        )
    else:
        prompt = (
            f"### 指示:\n{instruction}\n\n"
            f"### 回答:\n{output}"
        )
    
    # Rinnaモデル特有の改行コード変換
    prompt = prompt.replace('\n', '<NL>')
    
    # EOSトークンを付与
    prompt = prompt + tokenizer.eos_token
    
    # 
    return prompt

# 学習用と評価用に分割
train_val = dataset["train"].train_test_split(test_size=2000, shuffle=True, seed=42)
train_data = train_val["train"]
eval_data = train_val["test"]

#---------------------------------------------------------
# LoRAの設定
#---------------------------------------------------------
lora_config = LoraConfig(
    r=8, 
    lora_alpha=16,
    target_modules=["query_key_value"], 
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

#---------------------------------------------------------
# 学習引数の設定
#---------------------------------------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    learning_rate=3e-4,
    logging_steps=20,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=200,
    save_steps=200,
    save_total_limit=3,
    push_to_hub=False,
    report_to="none",
    
    # macOS (MPS) 最適化設定
    optim="adamw_torch",         
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=4,
    
    # 精度設定
    fp16=False,
    bf16=False,
    
    group_by_length=True,
)

#---------------------------------------------------------
# Trainerの準備 (SFTTrainer)
#---------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,  # 前回の修正点（ここも重要）
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=lora_config,
    formatting_func=formatting_prompts_func, # 修正した関数を使用
    #max_seq_length=CUTOFF_LEN,
    args=training_args,
)

#---------------------------------------------------------
# 学習の実行
#---------------------------------------------------------
print("学習を開始します...")
trainer.train()

#---------------------------------------------------------
# モデルの保存
#---------------------------------------------------------
trainer.model.save_pretrained(peft_name)
tokenizer.save_pretrained(peft_name)

print(f"学習完了: {peft_name} に保存しました。")