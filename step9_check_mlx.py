import sys
from mlx_lm import load, generate
from transformers import AutoTokenizer  # これを追加

#====================================================================
# 設定
#====================================================================
MODEL_PATH = "./models/mlx_model_fp16" # もしfp32で変換したならそのパス
ORIGINAL_MODEL_NAME = "ToPo-ToPo/ai-character-suuchi-kai-3.6b"

# Prompt
CHARACTER_SYSTEM_PROMPT = """
あなたは「解析カイ」という名前の新人アシスタントです。女の子です。
親しみやすいタメ口で会話します。
"""

#====================================================================
# モデルとトークナイザーの準備
#====================================================================
print("モデルを読み込んでいます...")

# 1. モデルは MLX で読み込む (高速化のため)
#    ※ここでの tokenizer は使わないので _ で捨てます
model, _ = load(MODEL_PATH)

# 2. トークナイザーは transformers で読み込む (正確性のため)
#    ※必ず use_fast=False を指定します
tokenizer = AutoTokenizer.from_pretrained(
    ORIGINAL_MODEL_NAME, 
    use_fast=False, 
    trust_remote_code=True
)

print("準備完了。")

#====================================================================
# 推論用関数の定義
#====================================================================
def generate_response(instruction, input_context=None):
    
    # プロンプト作成
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

    # 生成実行
    # 引数を正しく指定します
    response = generate(
        model, 
        tokenizer, 
        prompt=prompt,
        max_tokens=256,
        #temperature=0.85, # temp ではなく temperature に修正
        #top_p=0.9,
        #verbose=False
    )
    
    # 整形
    response = response.replace("<NL>", "\n")
    return response.strip()

#====================================================================
# 実行部分
#====================================================================
if __name__ == "__main__":
    
    questions = [
        "AITuberについて教えてください。",
        "日本で一番高い山はどこですか？",
        "美味しいカレーの作り方を教えて。",
        "まどか☆マギカでは誰が一番かわいい？"
    ]

    print("MLXモデル推論 (FP16 + SlowTokenizer)")
    print("-" * 50)
    for q in questions:
        print(f"質問: {q}")
        output = generate_response(instruction=q, input_context=CHARACTER_SYSTEM_PROMPT)
        print(f"回答: {output}")
        print("-" * 50)