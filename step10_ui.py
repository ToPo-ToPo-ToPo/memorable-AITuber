import torch
import time
import threading
import gradio as gr
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer
)

# ============================================================
# 設定
# ============================================================
BASE_MODEL_NAME = "ToPo-ToPo/ai-character-suuchi-kai-3.6b"

CHARACTER_SYSTEM_PROMPT = """
あなたは「解析カイ」という名前の新人アシスタントです。女の子です。
親しみやすいタメ口で会話します。
"""

STREAM_DELAY = 0.03          # 表示速度（秒 / トークン）
MAX_NEW_TOKENS = 256

# ============================================================
# モデル読み込み
# ============================================================
print("モデルを読み込んでいます...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_NAME,
    use_fast=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    device_map="mps",           # Apple Silicon
    low_cpu_mem_usage=True
)

model.eval()

# ウォームアップ（初回のカクつき防止）
dummy = tokenizer("test", return_tensors="pt").to(model.device)
model.generate(**dummy, max_new_tokens=1)

print("準備完了")

# ============================================================
# messages → モデル用プロンプト変換
# ============================================================
def build_prompt_from_messages(messages):
    """
    Gradio messages形式 → Rinna風プロンプト
    """
    prompt = f"### 入力:\n{CHARACTER_SYSTEM_PROMPT.strip()}\n\n"

    i = 0
    while i < len(messages) - 1:
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            prompt += f"### 指示:\n{messages[i]['content']}\n\n"
            prompt += f"### 回答:\n{messages[i + 1]['content']}\n\n"
            i += 2
        else:
            i += 1

    return prompt

# ============================================================
# ストリーミング生成（Gradio用）
# ============================================================
@torch.inference_mode()
def chat_stream(user_message, messages):
    """
    Gradio Chatbot (messages形式) 用ストリーミング関数
    """
    messages = messages or []

    # プロンプト構築
    prompt = build_prompt_from_messages(messages)
    prompt += f"### 指示:\n{user_message}\n\n### 回答:\n"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False
    ).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.85,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer
    )

    # 生成は別スレッド
    thread = threading.Thread(
        target=model.generate,
        kwargs=generation_kwargs
    )
    thread.start()

    # messages に新しい発言を追加
    messages = messages + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": ""}
    ]

    # ストリーミング表示
    for token in streamer:
        messages[-1]["content"] += token
        yield messages

        if token.endswith(("。", "！", "？", "\n")):
            time.sleep(STREAM_DELAY * 3)
        else:
            time.sleep(STREAM_DELAY)

# ============================================================
# Gradio UI
# ============================================================
with gr.Blocks(title="解析カイ Chatbot") as demo:
    gr.Markdown("## 🤖 解析カイ（Gradio 6.x / messages形式 / スムーズ表示）")

    chatbot = gr.Chatbot(
        height=450
    )

    textbox = gr.Textbox(
        placeholder="メッセージを入力してね",
        show_label=False
    )

    clear_btn = gr.Button("履歴クリア")

    textbox.submit(
        chat_stream,
        inputs=[textbox, chatbot],
        outputs=chatbot
    )

    textbox.submit(
        lambda: "",
        None,
        textbox
    )

    clear_btn.click(
        lambda: [],
        None,
        chatbot
    )

# ============================================================
# 起動
# ============================================================
if __name__ == "__main__":
    demo.queue().launch()