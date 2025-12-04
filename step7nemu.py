
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ====================================================================
# 1. 設定エリア
# ====================================================================
# ベースモデル
base_model_name = "ToPo-ToPo/rinna-japanese-gpt-neox-3.6b-lora-sft-v1"

# ★ここを変更: Phase 2で学習したLoRAモデルのパス
peft_name = "models/lora-rinna-3.6b-phase2-neum" 

# ★重要: 学習に使ったものと「全く同じ」システムプロンプトを貼り付けてください
# (これを入れないとキャラが降臨しません)
CHARACTER_SYSTEM_PROMPT = """
あなたは引きこもりゲーマーAITuberの「ネム」です。
性格は面倒くさがりでマイペース、常に眠そうです。
一人称は「ボク」、視聴者のことは「お前ら」や「リスナー」と呼びます。
口調は「〜だなぁ」「〜じゃね？」「ｗ」など、ネットスラング交じりの若者言葉を使います。
基本的にテンションは低めで、短文で返します。🎮や💤の絵文字を使います。
"""

# ====================================================================
# 2. モデルとトークナイザーの準備
# ====================================================================
print(f"モデルを読み込んでいます... ({peft_name})")

# ベースモデル読み込み (Mac/MPS用設定)
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

print("準備完了。AITuberを起動します！")

# ====================================================================
# 3. 推論用関数の定義 (Phase 2対応版)
# ====================================================================
def generate_response(user_input):
    """
    ユーザーの入力(user_input)を受け取り、
    システムプロンプトと結合してキャラクターとして回答させます。
    """
    
    # プロンプトの組み立て (Phase 2学習時のフォーマットを再現)
    # System -> ### 指示
    # User   -> ### 入力
    # Assistant -> ### 回答
    prompt = (
        f"### 指示:\n{CHARACTER_SYSTEM_PROMPT}\n\n"
        f"### 入力:\n{user_input}\n\n"
        f"### 回答:\n"
    )

    # トークナイズ
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        add_special_tokens=False
    ).to(model.device)

    input_token_len = inputs.input_ids.shape[1]

    # 生成実行
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.85, # キャラクター性を出すために少し高めに設定(0.7->0.85)
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # デコード
    generated_tokens = outputs[0][input_token_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # 後処理
    response = response.replace("<NL>", "\n")
    return response.strip()

# ====================================================================
# 4. 実行部分
# ====================================================================
if __name__ == "__main__":
    
    # キャラクターの反応を見るための質問リスト
    questions = [
        "自己紹介してください！",
        "今日の晩御飯なに食べた？",
        "最近あった嫌なこと教えて",
        "好きなお菓子はある？",
        "お前うるさいな",
        "トポロジー最適化ってなに？"
    ]

    print("-" * 50)
    print(f"[{peft_name}] 推論テスト")
    print("-" * 50)
    
    for q in questions:
        print(f"User: {q}")
        output = generate_response(q)
        print(f"AI  : {output}")
        print("-" * 50)