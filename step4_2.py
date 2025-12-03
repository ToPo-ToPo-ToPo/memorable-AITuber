
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#--------------------------------------------------------------------
# 1. モデルとトークナイザーの準備
#--------------------------------------------------------------------
model_name = "ToPo-ToPo/rinna-japanese-gpt-neox-3.6b"

print("モデルを読み込んでいます...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    use_fast=False
)
# パディングトークンの設定（エラー防止）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# MPSデバイスとfloat32を指定
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="mps",
    torch_dtype=torch.float32
)
print("準備完了。")

#--------------------------------------------------------------------
# 2. プロンプト作成関数 (LoRA学習時と同じフォーマット)
#--------------------------------------------------------------------
def create_prompt(instruction):
    prompt = (
        f"### 指示:\n{instruction}\n\n"
        f"### 回答:\n"
    )
    return prompt

#--------------------------------------------------------------------
# 3. 推論実行
#--------------------------------------------------------------------
def run_inference(question):
    # プロンプトの作成
    prompt = create_prompt(question)
    
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    
    # 入力トークンの長さを取得（後でカットするため）
    input_length = token_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            token_ids,
            max_new_tokens=128,      # 最大生成トークン数
            do_sample=True,          # ランダムサンプリング
            temperature=0.8,         # 創造性
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 入力部分を取り除いて、生成された続きだけをデコード
    generated_tokens = output_ids[0][input_length:]
    output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # 見やすく整形
    output = output.replace("<NL>", "\n")
    return output

#--------------------------------------------------------------------
# 実行
#--------------------------------------------------------------------
questions = [
    "まどか☆マギカでは誰が一番かわいい？",
    "日本で一番高い山はどこですか？"
]

for q in questions:
    print("-" * 50)
    print(f"指示: {q}")
    result = run_inference(q)
    print(f"生成結果: {result}")