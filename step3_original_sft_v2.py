from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
import transformers

#-------------------------------------------------------------------------------------
# Step 3: LoRAによるファインチューニング
# Rinnaフォーマット対応版
#-------------------------------------------------------------------------------------

# 基本情報
model_name = "ToPo-ToPo/rinna-japanese-gpt-neox-3.6b"
dataset = "kunishou/databricks-dolly-15k-ja"
peft_name = "models/lora-rinna-3.6b-v2"
output_dir = "models/lora-rinna-3.6b-v2-results"

#---------------------------------------------------------
# モデルの準備
#---------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
)

#---------------------------------------------------------
# トークナイザーの準備
#---------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# コンテキスト長
CUTOFF_LEN = 512 

# トークナイズ関数
def tokenize(prompt, tokenizer):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
    )
    return {
        "input_ids": result["input_ids"],
        "attention_mask": result["attention_mask"],
    }

#---------------------------------------------------------
# データセットの準備
#---------------------------------------------------------
data = load_dataset(dataset)

#---------------------------------------------------------
# 【修正箇所】プロンプト生成関数の変更
#---------------------------------------------------------
def generate_prompt(data_point):
    """
    Rinna形式のプロンプトを生成する関数 (タグなし・シンプル版)
    フォーマット:
    ユーザー: {入力}<NL>{指示}<NL>システム: {回答}
    """
    # 1. 改行変換
    instruction = data_point["instruction"].replace('\n', '<NL>')
    output = data_point["output"].replace('\n', '<NL>')
    
    # 2. ユーザーの発話を作成
    if data_point["input"]:
        input_context = data_point["input"].replace('\n', '<NL>')
        
        # 文脈(input)と指示(instruction)を、タグ無しで単に改行で繋ぎます
        # 順番: 文脈 -> 指示
        user_text = f"{input_context}<NL>{instruction}"
    else:
        user_text = instruction

    # 3. 全体を結合
    result = f"ユーザー: {user_text}<NL>システム: {output}"
    
    return result

#---------------------------------------------------------
# 学習データと検証データの準備
#---------------------------------------------------------
VAL_SET_SIZE = 2000

train_val = data["train"].train_test_split(
    test_size=VAL_SET_SIZE, 
    shuffle=True, 
    seed=42
)

# 教師データ
train_data = train_val["train"]
train_data = train_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))

# 評価用データ
val_data = train_val["test"]
val_data = val_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))

#---------------------------------------------------------
# LoRAモデルの準備
#---------------------------------------------------------
lora_config = LoraConfig(
    r=8, 
    lora_alpha=16,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

#---------------------------------------------------------
# トレーナーの準備
#---------------------------------------------------------
eval_steps = 200
save_steps = 200
logging_steps = 20

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        num_train_epochs=3,
        learning_rate=3e-4,
        logging_steps=logging_steps,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=save_steps,
        output_dir=output_dir,
        report_to="none",
        save_total_limit=3,
        push_to_hub=False,
        auto_find_batch_size=True
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(
        tokenizer, 
        mlm=False
    ),
)

#---------------------------------------------------------
# 学習の実行
#---------------------------------------------------------
model.config.use_cache = False
trainer.train() 
model.config.use_cache = True

#---------------------------------------------------------
# LoRAモデルの保存
#---------------------------------------------------------
trainer.model.save_pretrained(peft_name)