import torch
import time
import threading
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer
)

#====================================================================
# 設定エリア
#====================================================================
base_model_name = "ToPo-ToPo/ai-character-suuchi-kai-3.6b"

# システムプロンプト（キャラ設定）
CHARACTER_SYSTEM_PROMPT = """
あなたは「解析カイ」という名前の新人アシスタントです。女の子です。
親しみやすいタメ口で会話します。
"""

# 表示速度（秒 / トークン）
STREAM_DELAY = 0.05  # 0.02〜0.05 あたりがおすすめ

#====================================================================
# モデルとトークナイザーの準備
#====================================================================
print("モデルを読み込んでいます...")

tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    use_fast=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="mps",         # Apple Silicon
    low_cpu_mem_usage=True
)

model.eval()

print("準備完了。会話を開始します。")
print("-" * 60)

#====================================================================
# 推論用関数（スムーズ均等ストリーミング）
#====================================================================
@torch.inference_mode()
def generate_response_stream(
    instruction,
    input_context=None,
    delay=STREAM_DELAY
):
    """
    均等な速度でスムーズにストリーミング表示する
    """

    # プロンプト構築（Rinna形式）
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

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False
    ).to(model.device)

    # ★ IteratorStreamer（1トークンずつ取得）
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.85,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )

    # generate は別スレッドで実行
    generation_thread = threading.Thread(
        target=model.generate,
        kwargs=generation_kwargs
    )
    generation_thread.start()

    # 表示側（一定間隔）
    for text in streamer:
        print(text, end="", flush=True)

        # 句読点・改行で少し間を空けると自然
        if text.endswith(("。", "！", "？", "\n")):
            time.sleep(delay * 3)
        else:
            time.sleep(delay)

    print()  # 念のため改行

#====================================================================
# メイン実行部
#====================================================================
if __name__ == "__main__":

    questions = [
        "AITuberについて教えてください。",
        "日本で一番高い山はどこですか?",
        "美味しいカレーの作り方を教えて。",
        "まどか☆マギカでは誰が一番かわいい?"
    ]

    #------------------------------------------------------------
    # ウォームアップ（初回のカクつき防止）
    #------------------------------------------------------------
    print("ウォームアップ中（最初の1回は少し待ちます）...", end="", flush=True)
    dummy = tokenizer("test", return_tensors="pt").to(model.device)
    model.generate(**dummy, max_new_tokens=1)
    print("完了！\n")

    #------------------------------------------------------------
    # 本番ループ
    #------------------------------------------------------------
    for i, q in enumerate(questions, 1):
        print(f"Q{i}: {q}")
        print("A: ", end="", flush=True)

        generate_response_stream(
            instruction=q,
            input_context=CHARACTER_SYSTEM_PROMPT
        )

        print("-" * 60)