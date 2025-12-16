
#-------------------------------------------------------------------------
# 最適化版: 解析カイ Chatbot (Interactive Mode)
#-------------------------------------------------------------------------
import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

#====================================================================
# 設定
#====================================================================
base_model_name = "ToPo-ToPo/ai-character-suuchi-kai-3.6b"

# システムプロンプト（キャラクター設定）
CHARACTER_SYSTEM_PROMPT = """
あなたは「解析カイ」という名前の新人アシスタントです。女の子です。
親しみやすいタメ口で会話します。
ユーザーのことは「先輩」と呼びます。
"""

#====================================================================
# モデルとトークナイザーの準備
#====================================================================
print("===" * 20)
print("システムを起動しています... (モデル読み込み中)")
print("===" * 20)

# デバイスの自動判定 (Mac: mps, NVIDIA: cuda, その他: cpu)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"使用デバイス: {device}")

# 1. モデルの読み込み
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map=device,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
model.eval()

# 2. トークナイザーの読み込み
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("準備完了！会話を開始できます。")
print("終了するには 'exit' または 'quit' と入力してください。")
print("-" * 50)

#====================================================================
# 推論用関数の定義
#====================================================================
@torch.inference_mode()
def generate_response(instruction, input_context=None):
    """
    質問(instruction)を受け取り、LLMの回答を返します。
    """
    # プロンプト作成
    # キャラクター設定(input_context)を常に入力として渡すことで人格を維持します
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
    
    # トークナイズ
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        add_special_tokens=False
    ).to(model.device)
    
    input_token_len = inputs.input_ids.shape[1]
    
    # 生成実行
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.85,    # 創造性の調整
        top_p=0.9,
        repetition_penalty=1.1, # 繰り返し防止
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    
    # デコード
    generated_tokens = outputs[0][input_token_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # <NL>タグの処理
    response = response.replace("<NL>", "\n")
    
    return response.strip()

#====================================================================
# メインループ（チャット機能）
#====================================================================
def main():
    # 初回ウォームアップ（最初の1回は遅いため、空打ちしておく）
    print("ウォームアップ中...", end="\r")
    _ = generate_response("こんにちは", CHARACTER_SYSTEM_PROMPT)
    print(" " * 20, end="\r") # 表示クリア

    print("カイ: 先輩、お疲れ様です！何か手伝うことある？")

    while True:
        try:
            # ユーザー入力の受付
            user_input = input("\nあなた > ").strip()

            # 終了判定
            if user_input.lower() in ["exit", "quit", "終了", "バイバイ"]:
                print("\nカイ: お疲れ様でした！またね！")
                break
            
            # 空入力の無視
            if not user_input:
                continue

            # 応答生成中...
            print("...", end="", flush=True) 
            
            # AI応答の取得
            response = generate_response(
                instruction=user_input, 
                input_context=CHARACTER_SYSTEM_PROMPT
            )
            
            # 行頭の「...」を消して回答を表示
            print(f"\r\033[Kカイ > {response}")

        except KeyboardInterrupt:
            # Ctrl+C で強制終了された場合
            print("\n\nカイ: 強制終了？了解です。お疲れ様！")
            sys.exit()
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")

if __name__ == "__main__":
    main()