
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, TaskType
from trl import SFTTrainer

#-------------------------------------------------------------------------------------
# 1. 設定項目
#-------------------------------------------------------------------------------------
model_name = "ToPo-ToPo/rinna-japanese-gpt-neox-3.6b"

# 作成したローカルのJSONLファイルパスを指定
dataset_file = "outputs/character_data_mix_number_Nemu.jsonl"

# 保存先
peft_name = "models/trl-lora-rinna-3.6b-nemu-mps"
output_dir = "models/trl-lora-rinna-3.6b-nemu-results"

#---------------------------------------------------------
# 2. モデルの準備 (Float32 / MPS)
#---------------------------------------------------------
# macOSのMPS(Metal Performance Shaders)向け設定
# VRAM消費は増えますが、Float32の方がMPSでの動作は安定します
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="mps",         
    torch_dtype=torch.float32 
)
model.config.use_cache = False

#---------------------------------------------------------
# 3. トークナイザーの準備
#---------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

#---------------------------------------------------------
# 4. データセットの準備とフォーマット関数
#---------------------------------------------------------
# ローカルのJSONLファイルを読み込む
dataset = load_dataset("json", data_files=dataset_file)

# Phase 2用のフォーマット関数（Messages形式 → Rinnaテキスト形式）
def formatting_prompts_func(example):
    # datasetの各行（example）には "messages" リストが入っています
    messages = example['messages']
    
    system_text = ""
    user_text = ""
    assistant_text = ""

    # roleごとに内容を抽出
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        
        if role == "system":
            system_text = content
        elif role == "user":
            user_text = content
        elif role == "assistant":
            assistant_text = content

    # Rinna形式に整形 (改行は <NL> に置換)
    # Systemプロンプトも学習に含める構成にします
    prompt = (
        f"System: {system_text}<NL>"
        f"User: {user_text}<NL>"
        f"Assistant: {assistant_text}"
    )
    
    # 念のため本文中の改行もRinna仕様に置換
    # (ただしSystemなどのタグ部分は壊さないよう注意が必要ですが、
    #  今回は単純置換でも動作する範囲です)
    prompt = prompt.replace('\n', '<NL>')
    
    # EOSトークンを付与
    prompt = prompt + tokenizer.eos_token
    
    return [prompt] # SFTTrainerはリスト形式の戻り値を期待する場合があるためリスト化

# 学習用と評価用に分割
# データ総数が400件程度なので、test_sizeは小さく設定します（2000にするとエラーになります）
train_val = dataset["train"].train_test_split(test_size=20, shuffle=True, seed=42)
train_data = train_val["train"]
eval_data = train_val["test"]

#---------------------------------------------------------
# 5. LoRAの設定 (Phase 2 強力版)
#---------------------------------------------------------
lora_config = LoraConfig(
    r=32,                   # キャラクター性を出すため 8 -> 32
    lora_alpha=64,          # 重みを強めるため 16 -> 64
    target_modules=[        # 学習範囲を全線形層に拡大
        "query_key_value", 
        "dense", 
        "dense_h_to_4h", 
        "dense_4h_to_h"
    ], 
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

#---------------------------------------------------------
# 6. 学習引数の設定
#---------------------------------------------------------
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=10,       # データが少ないため多めに回す (3 -> 10)
    learning_rate=3e-4,
    logging_steps=10,
    eval_strategy="epoch",     # ステップごとではなくエポックごとに評価
    save_strategy="epoch",
    save_total_limit=2,
    push_to_hub=False,
    report_to="none",
    
    # macOS (MPS) 最適化設定
    optim="adamw_torch",         
    per_device_train_batch_size=1, # メモリ不足ならここを1のまま維持
    gradient_accumulation_steps=4, # 実質バッチサイズ = 1 * 4 = 4
    
    # Float32強制 (MPS安定性のため)
    fp16=False,
    bf16=False,
    
    group_by_length=True,
)

#---------------------------------------------------------
# 7. Trainerの準備 (SFTTrainer)
#---------------------------------------------------------
# dataset_text_fieldを指定せず、formatting_funcを使用する場合の構成
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=lora_config,
    formatting_func=formatting_prompts_func,
    args=training_args,
    max_length=512, # コンテキスト長を指定（重要）
)

#---------------------------------------------------------
# 8. 学習の実行
#---------------------------------------------------------
print(f"学習データ数: {len(train_data)}")
print("学習を開始します...")
trainer.train()

#---------------------------------------------------------
# 9. モデルの保存
#---------------------------------------------------------
trainer.model.save_pretrained(peft_name)
tokenizer.save_pretrained(peft_name)

print(f"学習完了: {peft_name} に保存しました。")