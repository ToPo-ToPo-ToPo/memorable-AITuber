#-------------------------------------------------------------------------
# 解析カイ Chatbot (会話履歴・<NL>トークン対応版)
#-------------------------------------------------------------------------
import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

#====================================================================
# 設定
#====================================================================
base_model_name = "ToPo-ToPo/ai-character-suuchi-kai-3.6b"
MAX_HISTORY = 10  # 記憶する会話の往復数

# システムプロンプト内の改行も <NL> に統一して、モデルの学習分布に合わせます
CHARACTER_SYSTEM_PROMPT = (
    "あなたは「解析カイ」という名前の新人アシスタントです。女の子です。<NL>"
    "親しみやすいタメ口で会話します。<NL>"
    "ユーザーのことは「先輩」と呼びます。"
)

#====================================================================
# モデルとトークナイザーの準備
#====================================================================
print("===" * 20)
print("システムを起動しています...")
print("===" * 20)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"使用デバイス: {device}")

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map=device,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("準備完了。<NL>対応モードで開始します。")
print("-" * 50)

#====================================================================
# プロンプト作成関数 (<NL>仕様)
#====================================================================
def build_prompt(history, current_user_input):
    """
    ユーザー指定の <NL> 連結ロジックに基づきプロンプトを構築します。
    """
    # 1. 履歴リストを文字列リストに変換
    # 例: ["ユーザー: こんにちは", "カイ: 先輩、こんにちは！"]
    formatted_turns = []
    
    # 過去ログ
    for turn in history:
        formatted_turns.append(f"{turn['speaker']}: {turn['text']}")
    
    # 今回のユーザー発言を追加
    formatted_turns.append(f"ユーザー: {current_user_input}")
    
    # 2. <NL> で結合
    # リストの中身を <NL> でつなぐ
    history_str = "<NL>".join(formatted_turns)
    
    # 3. 最終的なプロンプトの組み立て
    # システムプロンプト + <NL> + 会話履歴 + <NL> + カイ: 
    prompt = (
        f"{CHARACTER_SYSTEM_PROMPT}<NL>"
        f"{history_str}<NL>"
        f"カイ: "
    )
    return prompt

#====================================================================
# 推論実行
#====================================================================
@torch.inference_mode()
def generate_response_with_history(history, user_input):
    # プロンプト作成
    prompt = build_prompt(history, user_input)
    
    # デバッグ用に実際のプロンプトを確認したい場合はコメントアウトを外してください
    # print(f"\n[DEBUG PROMPT]: {prompt}\n")

    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        add_special_tokens=False
    ).to(model.device)
    
    input_token_len = inputs.input_ids.shape[1]

    # コンテキスト長対策（簡易版）
    if input_token_len > 1500:
        pass # 必要に応じて履歴削除処理を入れる

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.85,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    
    # デコード
    generated_tokens = outputs[0][input_token_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # 【重要】学習データに合わせて <NL> でカットする
    # モデルは "カイ: 回答<NL>ユーザー: " と予測する傾向があるため、最初の <NL> までを採用します。
    if "<NL>" in response:
        response = response.split("<NL>")[0]
    
    # 表示用に念のため普通の改行もチェック（モデルが混同した場合の保険）
    if "\n" in response:
        response = response.split("\n")[0]

    return response.strip()

#====================================================================
# メインループ
#====================================================================
def main():
    conversation_history = []

    # ウォームアップ
    print("ウォームアップ中...", end="\r")
    # ウォームアップでも空の履歴で実行
    _ = generate_response_with_history([], "テスト")
    print(" " * 20, end="\r")

    print("カイ: 先輩、お疲れ様です！準備万端ですよ！")

    while True:
        try:
            user_input = input("\nあなた > ").strip()

            if user_input.lower() in ["exit", "quit", "終了"]:
                print("\nカイ: お疲れ様でした！")
                break
            
            if not user_input:
                continue

            print("...", end="", flush=True) 
            
            response = generate_response_with_history(conversation_history, user_input)
            
            # 結果表示（ターミナルで見やすいように、表示時だけ <NL> があれば改行に直す）
            display_response = response.replace("<NL>", "\n")
            print(f"\r\033[Kカイ > {display_response}")

            # 履歴の更新
            # ここでは純粋なテキストのみ保存し、build_promptでspeakerを付与します
            conversation_history.append({"speaker": "ユーザー", "text": user_input})
            conversation_history.append({"speaker": "カイ", "text": response})

            if len(conversation_history) > MAX_HISTORY * 2:
                conversation_history = conversation_history[2:]

        except KeyboardInterrupt:
            print("\n\nカイ: 強制終了します！")
            sys.exit()
        except Exception as e:
            print(f"\nエラー: {e}")

if __name__ == "__main__":
    main()