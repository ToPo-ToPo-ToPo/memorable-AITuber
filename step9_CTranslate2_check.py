import ctranslate2
import transformers

#====================================================================
# 設定
#====================================================================
# 変換したCTranslate2モデルのフォルダパス
ct2_model_dir = "./models/ai-character-suuchi-kai-3.6b-Ctranslate2"

# トークナイザー用の元のモデル名（またはパス）
base_model_name = "ToPo-ToPo/ai-character-suuchi-kai-3.6b"

# システムプロンプト（PyTorch版と同じ）
CHARACTER_SYSTEM_PROMPT = """
あなたは「解析カイ」という名前の新人アシスタントです。女の子です。
親しみやすいタメ口で会話します。
"""

#====================================================================
# モデルとトークナイザーの準備
#====================================================================
print("モデルを読み込んでいます(CTranslate2)...")

# 1. ジェネレーターの読み込み
# Macの場合、CTranslate2はCPU実行が基本ですが、非常に高速に最適化されています。
# device="cuda" はNVIDIA GPU用です。Macでは "cpu" を指定します。
generator = ctranslate2.Generator(ct2_model_dir, device="cpu")

# 2. トークナイザーの読み込み
tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_name, use_fast=True)

print("準備完了。")

#====================================================================
# 推論用関数の定義
#====================================================================
def generate_response(instruction, input_context=None):
    """
    CTranslate2を使用して回答を生成します。
    """
    # 1. プロンプトの作成（PyTorch版と完全に一致させる）
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
    # CTranslate2は「トークンの文字列リスト」を入力として受け取ります
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # 3. 生成実行
    # PyTorch版のパラメータに合わせて調整
    results = generator.generate_batch(
        [input_tokens],
        max_length=256,          # PyTorch: max_new_tokens=256
        sampling_topk=0,         # PyTorchでは指定なし(0で無効化)
        sampling_topp=0.9,       # PyTorch: top_p=0.9
        sampling_temperature=0.85, # PyTorch: temperature=0.85
        repetition_penalty=1.1,  # PyTorch: repetition_penalty=1.1
        include_prompt_in_result=False # 結果にプロンプトを含めない
    )

    # 4. デコード
    # 生成されたトークン文字列をIDに戻してからデコードすることで、綺麗なテキストにします
    output_tokens = results[0].sequences_ids[0]
    response = tokenizer.decode(output_tokens, skip_special_tokens=True)

    # Rinna特有の <NL> を改行に戻す
    response = response.replace("<NL>", "\n")

    return response.strip()

#====================================================================
# 実行部分
#====================================================================
if __name__ == "__main__":
    questions = [
        "AITuberについて教えてください。",
        "日本で一番高い山はどこですか?",
        "美味しいカレーの作り方を教えて。",
        "まどか☆マギカでは誰が一番かわいい?"
    ]

    print("CTranslate2 (INT8) 推論モデル")
    print("-" * 50)

    # ウォームアップ
    print("ウォームアップ中...")
    _ = generate_response("テスト", CHARACTER_SYSTEM_PROMPT)
    print()

    for q in questions:
        print(f"質問: {q}")
        output = generate_response(instruction=q, input_context=CHARACTER_SYSTEM_PROMPT)
        print(f"回答: {output}")
        print("-" * 50)