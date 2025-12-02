
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
#-------------------------------------------------------------------------------------
# Step 3: LoRAによるファインチューニング
#-------------------------------------------------------------------------------------
# 基本情報
model_name = "ToPo-ToPo/rinna-japanese-gpt-neox-3.6b"
dataset = "kunishou/databricks-dolly-15k-ja"
peft_name = "lora-rinna-3.6b"
output_dir = "lora-rinna-3.6b-results"

#---------------------------------------------------------
# モデルの準備
#---------------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
)

#---------------------------------------------------------
#トークナイザーの準備
#---------------------------------------------------------
# 設定
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

CUTOFF_LEN = 256  # コンテキスト長

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
# 使用するデータセットを取得
data = load_dataset(dataset)

# データセットの確認
data["train"][5]

# 学習用のレスポンス版のプロンプトを定義(質問の回答を生成させる)
def generate_prompt(data_point):

    if data_point["input"]:
        result = (
            f"### 指示:\n"
            f"{data_point['instruction']}\n\n"
            f"### 入力:\n"
            f"{data_point['input']}\n\n"
            f"### 回答:\n"
            f"{data_point['output']}"
        )
    else:
        result = (
            f"### 指示:\n"
            f"{data_point['instruction']}\n\n"
            f"### 回答:\n"
            f"{data_point['output']}"
        )

    # 改行→<NL>
    result = result.replace('\n', '<NL>')
    return result

# 学習用のプロンプトテンプレートの確認
print(generate_prompt(data_point=data["train"][5]))

# 学習データと検証データの準備
VAL_SET_SIZE = 2000

# 学習データと検証データの準備
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
# LoRAのパラメータ
lora_config = LoraConfig(
    r=8, 
    lora_alpha=16,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# LoRAモデルの準備
model = get_peft_model(model, lora_config)

# 学習可能パラメータの確認
model.print_trainable_parameters()

#---------------------------------------------------------
# トレーナーの準備
#---------------------------------------------------------
import transformers
eval_steps = 200
save_steps = 200
logging_steps = 20

# トレーナーの準備
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=transformers.TrainingArguments(
        num_train_epochs=3,
        learning_rate=3e-4,
        logging_steps=logging_steps,
        evaluation_strategy="steps",
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