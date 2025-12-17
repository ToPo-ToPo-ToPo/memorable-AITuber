import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

#====================================================================
# 設定
#====================================================================
# ベースモデル（学習に使ったものと同じ）
base_model_name = "ToPo-ToPo/rinna-japanese-gpt-neox-3.6b"

# 学習したLoRAアダプターの保存先フォルダ
# ※前回保存したフォルダ名に合わせてください
peft_name = "models/lora-rinna-3.6b-v2" 

#====================================================================
# モデルとトークナイザーの準備
#====================================================================
print("モデルを読み込んでいます...")

# 1. ベースモデルの読み込み (MPS / float32)
# Macの場合、float32が最も安定します
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="mps",
    torch_dtype=torch.float32,
)

# 2. トークナイザーの読み込み
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. LoRAアダプターをベースモデルに結合
model = PeftModel.from_pretrained(
    base_model, 
    peft_name,
    device_map="mps"
)

# 推論モードに設定
model.eval()

print("準備完了。")

#====================================================================
# 推論用関数の定義 (Rinnaフォーマット対応版)
#====================================================================
def generate_response(instruction, input_context=None):
    """
    質問(instruction)を受け取り、LLMの回答を返します。
    """
    
    # 1. 改行を <NL> に変換 (Rinna仕様)
    instruction = instruction.replace("\n", "<NL>")
    
    # 2. プロンプトの作成
    # フォーマット: ユーザー: {文脈}<NL>{指示}<NL>システム: 
    if input_context:
        input_context = input_context.replace("\n", "<NL>")
        # 文脈と指示を結合してユーザー発話にする
        user_text = f"{input_context}<NL>{instruction}"
    else:
        user_text = instruction

    # システムへの入力プロンプト完成形
    prompt = f"ユーザー: {user_text}<NL>システム: "

    # 3. トークナイズ
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        add_special_tokens=False
    ).to(model.device)

    # 入力したトークンの長さを記憶 (回答部分だけを取り出すため)
    input_token_len = inputs.input_ids.shape[1]

    # 4. 生成実行
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 5. デコード
    # 入力部分以降のトークンのみをデコード
    generated_tokens = outputs[0][input_token_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # <NL> を改行に戻して表示しやすくする
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

    print(f"LoRAモデル確認: {peft_name}")
    print("-" * 50)
    for q in questions:
        print(f"ユーザー: {q}")
        # 生成実行
        output = generate_response(q)
        print(f"システム: {output}")
        print("-" * 50)