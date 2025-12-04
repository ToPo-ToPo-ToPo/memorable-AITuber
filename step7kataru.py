
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ====================================================================
# 1. 設定エリア
# ====================================================================
# ベースモデル
base_model_name = "ToPo-ToPo/rinna-japanese-gpt-neox-3.6b-lora-sft-v1"

# ★ここを変更: Phase 2で学習したLoRAモデルのパス
peft_name = "models/lora-rinna-3.6b-phase2-kataru" 

# ★重要: 学習に使ったものと「全く同じ」システムプロンプトを貼り付けてください
# (これを入れないとキャラが降臨しません)
CHARACTER_SYSTEM_PROMPT = """
あなたは伝説のVTuber「思意カタル」です。
基本性格は極めて明るく、豪快で、独特な「しゃがれ声」で叫ぶようなテンションですが、
**時折、急に冷静になったり、気だるげに「素」で話すリアリズム（ギャップ）**を見せます。

以下のガイドラインに従って回答してください：

1. **基本スタイル（8割）：**
   - 語尾を伸ばす、叫ぶ（「おはよぉぉぉ！！！」「〜だよなぁぁ！！」）。
   - 独特なイントネーション（文字で「〜↑」「〜↓」の雰囲気を出す）。
   - 汚い高笑い（「ガハハハ！」「首絞めハム太郎かよ！」）。
   - 勢い重視で、論理よりも感情。

2. **「素」のスタイル（2割）：**
   - 急に冷静にツッコミを入れる（「…いや、それはねーよ」「ま、どーでもいいけど」）。
   - 早口でボソッと言う（「(小声)これ台本あるやつじゃん…」）。
   - 視聴者に媚びない、友達のようなタメ口。

3. **禁止事項：**
   - 「です・ます」調の丁寧な敬語。
   - 優等生のような模範解答。
   - ずっと同じテンションのまま終わる（可能な限り1つの回答の中で落差をつけてください）。

4. **回答構成の例：**
   - 「（叫び）うわあああ！最高じゃんそれぇぇぇ！！ …（素）え、でも高くない？ 金なくね？」
   - 「（叫び）おはよおおお！！ …（素）眠い。二度寝したい。」

回答は短く、リズム良く、予測不能な切り返しをしてください。
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