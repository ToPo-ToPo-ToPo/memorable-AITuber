import torch
import sys
import threading
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TextIteratorStreamer
)

# ==========================================
# 1. 設定セクション
# ==========================================

# モデル名
MODEL_NAME = "ToPo-ToPo/ai-character-suuchi-kai-3.6b-v2"

# システムプロンプト（キャラクター設定）
CHARACTER_SYSTEM_PROMPT = """
あなたは「数値カイ」という名前の新人アシスタントです。女の子です。
親しみやすいタメ口で会話します。
""".strip()

# 生成パラメータ
MAX_NEW_TOKENS = 512  # 生成する最大トークン数
MEMORY_TURNS = 2      # 記憶する過去の会話往復数 (0にすると記憶なし)
TEMPERATURE = 0.7     # 創造性 (0.1〜1.0)
TOP_P = 0.9           # 分布の切り捨て

# ==========================================
# 2. 初期化・モデルロード
# ==========================================

print("===" * 20)
print(f"システム起動中: {MODEL_NAME}")
print("===" * 20)

# デバイス設定
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
    print("使用デバイス: CUDA (GPU)")
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float32 # Mac用 (float16は不安定な場合があるため32推奨)
    print("使用デバイス: MPS (Mac Silicon)")
else:
    device = "cpu"
    dtype = torch.float32
    print("使用デバイス: CPU")

# トークナイザーとモデルのロード
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=device,
    torch_dtype=dtype,
)

print(f"準備完了。過去 {MEMORY_TURNS} 回分の会話を記憶します。")
print("-" * 50)

# ==========================================
# 3. ユーティリティ関数
# ==========================================

def clean_response(text):
    """
    生成されたテキストが文の途中で終わっている場合、
    最後の句読点までカットして整える関数。
    """
    # 終了とみなす文字
    ends = ["。", "！", "？", "\n", "!", "?", "."]
    
    stripped_text = text.strip()
    
    # すでにきれいに終わっていればそのまま返す
    if any(stripped_text.endswith(e) for e in ends):
        return stripped_text

    # 後ろからスキャンして、最初の句読点を探す
    for i in range(len(stripped_text) - 1, -1, -1):
        if stripped_text[i] in ends:
            # 句読点まででカットして返す
            return stripped_text[:i+1]
    
    # 句読点が一つもない場合は、そのまま返す（あるいは空文字にする判断も可）
    return stripped_text

def build_prompt(current_input, history):
    """
    プロンプト構築関数
    - 過去の履歴を MEMORY_TURNS 分だけ取得
    - <NL> フォーマットに変換
    - 最新の入力にはシステムプロンプトを付与
    """
    prompt = ""

    # 1. 履歴の追加 (0の場合はスキップ)
    if MEMORY_TURNS > 0:
        # リストの後ろからN個を取得
        recent_history = history[-MEMORY_TURNS:]
    else:
        recent_history = []

    for past_user, past_bot in recent_history:
        safe_past_user = past_user.replace("\n", "<NL>")
        safe_past_bot = past_bot.replace("\n", "<NL>")
        
        # 過去ログ: ユーザー: {発言}<NL>システム: {回答}<NL>
        prompt += f"ユーザー: {safe_past_user}<NL>システム: {safe_past_bot}<NL>"

    # 2. 今回の入力を追加 (システムプロンプト付き)
    # フォーマット: ユーザー: {設定}<NL>{発言}<NL>システム: 
    safe_current_input = current_input.replace("\n", "<NL>")
    safe_system = CHARACTER_SYSTEM_PROMPT.replace("\n", "<NL>")
    
    prompt += f"ユーザー: {safe_system}<NL>{safe_current_input}<NL>システム: "
    
    return prompt

def generate_stream(current_input, history):
    """
    ストリーミング生成を行うジェネレーター
    """
    prompt = build_prompt(current_input, history)

    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        add_special_tokens=False
    ).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=10.0
    )

    generation_kwargs = dict(
        input_ids=inputs.input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    generated_text = ""
    
    # ストリーミングループ
    for new_text in streamer:
        generated_text += new_text
        
        # 表示用に<NL>を改行に変換してyield
        display_chunk = new_text.replace("<NL>", "\n")
        yield display_chunk

    return generated_text.replace("<NL>", "\n")

# ==========================================
# 4. メイン実行ループ
# ==========================================

def main():
    # 会話履歴保存用: [(user, bot), (user, bot), ...]
    history = []

    print("カイ: ボス、お疲れ様！何か手伝うことある？")

    while True:
        try:
            user_input = input("\nあなた > ").strip()

            # 終了判定
            if user_input.lower() in ["exit", "quit", "終了"]:
                print("\nカイ: お疲れ様でした！またね！")
                break
            
            if not user_input:
                continue

            print("カイ > ", end="", flush=True)
            
            full_response = ""
            
            # --- 生成実行 ---
            stream_generator = generate_stream(user_input, history)
            
            for chunk in stream_generator:
                print(chunk, end="", flush=True)
                full_response += chunk
            
            # 末尾の改行確保
            if not full_response.endswith("\n"):
                print()

            # --- 後処理 (文末カット) ---
            clean_text = clean_response(full_response)
            
            # カットが発生した場合の通知（デバッグ用、不要なら削除可）
            if len(clean_text) < len(full_response.strip()):
                # print("(※文末を調整しました)") 
                pass

            # --- 履歴更新 ---
            # 次回のプロンプトには「きれいな状態のテキスト」を渡す
            history.append((user_input, clean_text))
            
            # 履歴リスト自体が肥大化しないように制限（プロンプト作成とは別に、Pythonメモリ管理用）
            if len(history) > 20:
                history = history[-5:]

        except KeyboardInterrupt:
            print("\n\nカイ: 終了します！")
            sys.exit()
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")

if __name__ == "__main__":
    main()