import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ====================================================================
# 1. 設定エリア
# ====================================================================
# ベースモデル
base_model_name = "ToPo-ToPo/rinna-japanese-gpt-neox-3.6b-lora-sft-v2"

# Phase 2で学習したLoRAモデルのパス
peft_name = "models/rinna-japanese-gpt-neox-3.6b-lora-suuchi-kai-v2"

CHARACTER_SYSTEM_PROMPT = """
あなたは「数値カイ」という名前の新人アシスタントです。女の子です。
親しみやすいタメ口で会話します。
"""

# ====================================================================
# 2. モデルとトークナイザーの準備
# ====================================================================
print(f"モデルを読み込んでいます... ({peft_name})")

# ベースモデル読み込み (Mac/MPS用)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="mps",
    torch_dtype=torch.float32,
)

# トークナイザー読み込み
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# LoRAアダプター結合
model = PeftModel.from_pretrained(
    model, 
    peft_name,
    device_map="mps"
)
model.eval()

print("準備完了。純粋なLoRAモデルとして起動します。")

# ====================================================================
# 3. 推論用関数の定義 (シンプル版)
# ====================================================================
def generate_response(user_input):
    """
    ユーザーの入力(user_input)に対し、
    LoRAで学習したキャラクター性のみで回答させます。
    """
    
    # 1. 改行コードの変換 (Rinna仕様)
    user_input = user_input.replace("\n", "<NL>")
    
    # 2. プロンプトの組み立て
    # 余計な指示は入れず、ユーザーの言葉だけを渡します。
    # フォーマット: ユーザー: {発言}<NL>システム: 
    
    prompt = f"ユーザー: {CHARACTER_SYSTEM_PROMPT}<NL>{user_input}<NL>システム: "

    # 3. トークナイズ
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        add_special_tokens=False
    ).to(model.device)

    input_token_len = inputs.input_ids.shape[1]

    # 4. 生成実行
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7, # 素の性能を見るため、標準的な値に戻しました
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 5. デコード
    generated_tokens = outputs[0][input_token_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # 後処理
    response = response.replace("<NL>", "\n")
    return response.strip()

# ====================================================================
# 4. 実行部分
# ====================================================================
if __name__ == "__main__":
    
    questions = [
        "自己紹介してください！",  # ここで名前を名乗れるかが成功の鍵です
        "今日の晩御飯なに食べた？",
        "最近あった嫌なこと教えて",
        "好きなお菓子はある？",
        "お前うるさいな",
        "トポロジー最適化ってなに？"
    ]

    print("-" * 50)
    print(f"[{peft_name}] 推論テスト (System Promptなし)")
    print("-" * 50)
    
    for q in questions:
        print(f"User: {q}")
        output = generate_response(q)
        print(f"AI  : {output}")
        print("-" * 50)