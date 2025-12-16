
#-------------------------------------------------------------------------
# Step5-2: 学習済みのLoRAアダプターをマージしたモデルの推論
#-------------------------------------------------------------------------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
#====================================================================
# 設定
#====================================================================
# ベースモデル（学習に使ったものと同じ）
base_model_name = "models/rinna-japanese-gpt-neox-3.6b-lora-suuchi-kai"

# Prompt
CHARACTER_SYSTEM_PROMPT = """
あなたは「解析カイ」という名前の新人アシスタントです。女の子です。
親しみやすいタメ口で会話します。
"""
#====================================================================
# モデルとトークナイザーの準備
#====================================================================
print("モデルを読み込んでいます...")

# 1. モデルの読み込み (MPS / float32)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="mps",
    torch_dtype=torch.float32,
)

# 2. トークナイザーの読み込み
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


print("準備完了。")

#====================================================================
# 推論用関数の定義
#====================================================================
def generate_response(instruction, input_context=None):
    """
    質問(instruction)を受け取り、LLMの回答を返します。
    """
    
    # 1. プロンプトの作成
    if input_context:
        prompt = (
            f"### 指示:\n{instruction}\n\n"
            f"### 入力:\n{input_context}\n\n"
            f"### 回答:\n"
        )
    else:
        prompt = (
            f"### 指示:\n{instruction}\n\n"
            f"### 回答:\n"
        )

    # 2. トークナイズ
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        add_special_tokens=False
    ).to(model.device)

    # 入力したトークンの長さを記憶しておく
    input_token_len = inputs.input_ids.shape[1]

    # 3. 生成実行
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.85,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 4. デコード
    # 出力された全トークン(outputs[0])から、
    # 入力トークン分(input_token_len)より後ろだけを取り出してデコード
    generated_tokens = outputs[0][input_token_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Rinna特有の <NL> を改行に戻す
    response = response.replace("<NL>", "\n")
    
    return response.strip()

#====================================================================
# 実行部分
#====================================================================
if __name__ == "__main__":
    
    # テスト用の質問リスト
    questions = [
        "AITuberについて教えてください。",
        "日本で一番高い山はどこですか？",
        "美味しいカレーの作り方を教えて。",
        "まどか☆マギカでは誰が一番かわいい？"
    ]

    print("LoRA学習済みモデル")
    print("-" * 50)
    for q in questions:
        print(f"質問: {q}")
        output = generate_response(instruction=q, input_context=CHARACTER_SYSTEM_PROMPT)
        print(f"回答: {output}")
        print("-" * 50)
    