
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

#-------------------------------------------------------------------------------------
# 基本情報
#-------------------------------------------------------------------------------------
model_name = "ToPo-ToPo/rinna-japanese-gpt-neox-3.6b"
dataset_name = "kunishou/databricks-dolly-15k-ja"
output_dir = "lora-rinna-3.6b-results-sft"

#---------------------------------------------------------
# モデルとトークナイザーの準備
#---------------------------------------------------------
# Rinnaは use_fast=False 必須
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token # パディングトークンの設定

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    #torch_dtype=torch.float16, # GPUのメモリ節約
    use_cache=False # 学習時はFalse
)

#---------------------------------------------------------
# データセットの準備
#---------------------------------------------------------
data = load_dataset(dataset_name, split="train")

# 学習データと検証データに分割
data = data.train_test_split(test_size=2000, seed=42)
train_data = data["train"]
val_data = data["test"]

#---------------------------------------------------------
# フォーマット関数 (SFTTrainer用)
#---------------------------------------------------------
# データセットのバッチを受け取り、テキストのリストを返す関数を作ります
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        # データ取得
        instruction = example['instruction'][i]
        input_text = example['input'][i]
        output = example['output'][i]

        # Rinna用プロンプト作成
        if input_text:
            text = f"### 指示:\n{instruction}\n\n### 入力:\n{input_text}\n\n### 回答:\n{output}"
        else:
            text = f"### 指示:\n{instruction}\n\n### 回答:\n{output}"
        
        # Rinna特有の <NL> 変換
        text = text.replace('\n', '<NL>')
        
        # EOSトークンを追加 (SFTTrainerは自動でつけない場合があるため明示)
        text += tokenizer.eos_token
        
        output_texts.append(text)
    return output_texts

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
# トレーナーの準備 (SFTTrainer)
#---------------------------------------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=4, # メモリに応じて調整
    gradient_accumulation_steps=4,
    learning_rate=3e-4,
    logging_steps=20,
    eval_strategy="steps", # versionによっては evaluation_strategy
    eval_steps=200,
    save_steps=200,
    save_total_limit=3,
    fp16=True, # GPUならTrue
    report_to="none",
    push_to_hub=False,
)

# 補足: DataCollatorForCompletionOnlyLM を使うと
# 「指示」部分のLossを計算せず、「回答」部分だけ学習させることができ、精度が上がります。
# Rinnaフォーマットに合わせて "### 回答:\n" 以降を学習対象にします。
# response_template = "### 回答:\n" 
# ※Rinnaは改行が <NL> なので、正確にはトークナイズ後のIDでマッチさせる等の調整が必要な場合がありますが、
# 単純化のため今回は標準のCollator（全体学習）でもOKです。
# ここではSFTTrainer標準の挙動（全体学習）で記述します。

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    peft_config=lora_config,
    formatting_func=formatting_prompts_func, # ここに整形関数を渡すだけ
    max_seq_length=512, # コンテキスト長
    tokenizer=tokenizer,
    args=training_args,
    packing=False, # Trueにすると高速化するが、短いデータが多いとpadding問題が出る場合も
)

#---------------------------------------------------------
# 学習の実行
#---------------------------------------------------------
print("学習開始...")
trainer.train()

# 保存
trainer.model.save_pretrained(output_dir)