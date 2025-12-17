import os
import random
import threading

import torch
import gradio as gr
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
)

# ==========================================
# 1. 設定
# ==========================================

MODEL_NAME = "rinna/japanese-gpt-neox-3.6b-instruction-sft-v2"
CHAR_IMAGE_DIR = "assets/characters"

SYSTEM_PROMPT = (
    "あなたはAITuberの『根夢（ねむ）』です。"
    "明るく親しみやすく、短めに会話してください。"
    "難しい言葉は使わず、自然な日本語で話します。"
)

MAX_NEW_TOKENS = 256

# ==========================================
# 2. デバイス判定
# ==========================================

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float32
else:
    device = "cpu"
    dtype = torch.float32

print(f"使用デバイス: {device}")

# ==========================================
# 3. モデルロード
# ==========================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=device,
    torch_dtype=dtype,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("モデル準備完了")

# ==========================================
# 4. ユーティリティ
# ==========================================

def normalize(content):
    """Gradio 6.x の content 正規化"""
    if isinstance(content, list):
        return "".join(str(x) for x in content)
    return content or ""

def get_random_character_image():
    images = [
        os.path.join(CHAR_IMAGE_DIR, f)
        for f in os.listdir(CHAR_IMAGE_DIR)
        if f.lower().endswith(".png")
    ]
    return random.choice(images) if images else None

# ==========================================
# 5. ストリーミング生成
# ==========================================

def stream_generate(message, history):
    prompt = f"システム: {SYSTEM_PROMPT}<NL>"

    paired_history = []
    current_user = None

    for item in history:
        role = item["role"]
        content = normalize(item["content"])

        if role == "user":
            current_user = content
        elif role == "assistant" and current_user is not None:
            paired_history.append((current_user, content))
            current_user = None

    for user_turn, bot_turn in paired_history[-3:]:
        prompt += (
            f"ユーザー: {user_turn.replace(chr(10), '<NL>')}<NL>"
            f"システム: {bot_turn.replace(chr(10), '<NL>')}<NL>"
        )

    prompt += f"ユーザー: {message.replace(chr(10), '<NL>')}<NL>システム: "

    input_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )

    thread = threading.Thread(
        target=model.generate,
        kwargs=generation_kwargs,
    )
    thread.start()

    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text.replace("<NL>", "\n")

# ==========================================
# 6. Gradio 応答関数
# ==========================================

def respond(message, history):
    character_image = get_random_character_image()

    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]

    generator = stream_generate(message, history)

    partial = ""
    for text in generator:
        partial = text
        new_history[-1]["content"] = partial
        yield new_history, character_image

# ==========================================
# 7. Gradio UI
# ==========================================

with gr.Blocks() as demo:
    gr.Markdown("## 🎭 AITuber Nemu (Proto)")

    with gr.Row():
        chatbot = gr.Chatbot(
            height=500,
        )
        character = gr.Image(
            label="Character",
            type="filepath",
            height=500,
        )

    msg = gr.Textbox(
        placeholder="メッセージを入力してください",
        show_label=False,
    )

    msg.submit(
        respond,
        inputs=[msg, chatbot],
        outputs=[chatbot, character],
    )

if __name__ == "__main__":
    demo.launch()