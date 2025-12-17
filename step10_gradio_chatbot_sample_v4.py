import os
import random
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

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

MAX_NEW_TOKENS = 64
MAX_SENTENCES = 5   # 1回の応答で最大何文話すか

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
# 5. 1文だけ生成する関数
# ==========================================

def generate_one_sentence(user_message, history):
    prompt = f"システム: {SYSTEM_PROMPT}\n"

    for item in history:
        role = item["role"]
        content = normalize(item["content"])
        if role == "user":
            prompt += f"ユーザー: {content}\n"
        else:
            prompt += f"システム: {content}\n"

    if user_message:
        prompt += f"ユーザー: {user_message}\n"

    prompt += "システム: "

    input_ids = tokenizer.encode(
        prompt,
        return_tensors="pt",
        add_special_tokens=False
    ).to(model.device)

    # 文末で止める
    eos_ids = [
        tokenizer.encode("。", add_special_tokens=False)[0],
        tokenizer.encode("！", add_special_tokens=False)[0],
        tokenizer.encode("？", add_special_tokens=False)[0],
    ]

    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=eos_ids,
        pad_token_id=tokenizer.pad_token_id,
    )

    text = tokenizer.decode(
        output[0][input_ids.shape[1]:],
        skip_special_tokens=True
    ).strip()

    return text

# ==========================================
# 6. Gradio 応答関数（文ごと処理）
# ==========================================

def respond(message, history):
    ui_history = list(history)
    ui_history.append({"role": "user", "content": message})

    for i in range(MAX_SENTENCES):
        sentence = generate_one_sentence(message if i == 0 else "", ui_history)

        if not sentence:
            break

        ui_history.append(
            {"role": "assistant", "content": sentence}
        )

        character_image = get_random_character_image()
        yield ui_history, character_image

def respond(message, history):
    # UI 用（表示専用）
    ui_history = list(history)
    ui_history.append({"role": "user", "content": message})
    ui_history.append({"role": "assistant", "content": ""})

    # LLM 用（生成専用・確定文のみ）
    llm_history = list(history)
    llm_history.append({"role": "user", "content": message})

    displayed_text = ""

    for i in range(MAX_SENTENCES):
        sentence = generate_one_sentence(
            user_message="" if i > 0 else message,
            history=llm_history
        )

        if not sentence:
            break

        # --- UI 表示 ---
        displayed_text += sentence
        ui_history[-1]["content"] = displayed_text

        # --- LLM 履歴には「1文だけ」追加 ---
        llm_history.append(
            {"role": "assistant", "content": sentence}
        )

        character_image = get_random_character_image()
        yield ui_history, character_image

# ==========================================
# 7. Gradio UI
# ==========================================

with gr.Blocks() as demo:
    gr.Markdown("## 🎭 AITuber Nemu（1文生成・画像切替版）")

    with gr.Row():
        chatbot = gr.Chatbot(height=500)
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