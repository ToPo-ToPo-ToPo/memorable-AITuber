import os
import re
import random
import threading
import time

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

MODEL_NAME = "ToPo-ToPo/rinna-japanese-gpt-neox-3.6b-lora-sft-v2"
CHAR_IMAGE_DIR = "assets/characters_v1"

SYSTEM_PROMPT = ""

MAX_NEW_TOKENS = 512
SENTENCE_PAUSE_DURATION = 1.5

DEFAULT_IMAGE_PATH = os.path.join(CHAR_IMAGE_DIR, "normal1.png")

# 【修正】CSS定義: 「バストアップ（上半身）」を表示する設定
# scale(1.8) で1.8倍にズームし、origin で顔の位置に合わせています。
CUSTOM_CSS = """
#character-view {
    height: 700px !important;
    overflow: hidden;         /* 拡大してはみ出した部分（下半身）を隠す */
    border: none !important;
}
#character-view img {
    width: 100%;
    height: 100%;
    object-fit: contain;           /* まずは比率を崩さず表示 */
    transform: scale(1.3);         /* そこから1.3倍に拡大（ここを数字変えれば調節可） */
    transform-origin: center 0%;  /* 拡大の基準点を「中央・上寄り（顔）」に設定 */
}
"""

# ==========================================
# 2. キーワードと画像の対応
# ==========================================

EMOTION_SETTINGS = {
    "joy": {
        "keywords": ["嬉しい", "楽しい", "わーい", "やった", "最高", "好き", "笑", "ハハ", "あはは"],
        "images": ["happy1.png", "happy2.png", "happy3.png", "normal2.png", "normal4.png", "normal9.png", "normal12.png"]
    },
    "sadness": {
        "keywords": ["悲しい", "つらい", "残念", "困", "泣", "うう", "ひどい", "ごめん", "悪い", "気分がよくない", "辛い"],
        "images": ["sad1.png", "sad2.png"]
    },
    "anger": {
        "keywords": ["怒", "ムカ", "イライラ", "許せない", "激怒", "ぷんぷん", "最低"],
        "images": ["angry1.png", "angry2.png"]
    },
    "lazy": {
        "keywords": ["めんどい", "めんどくさい", "えー。", "大変", "あー。", "ヤダヤダ", "やりたくない"],
        "images": ["lazy1.png", "lazy2.png", "lazy3.png", "lazy4.png"]
    },
    "thinking": {
        "keywords": ["考え", "うーん", "どうしよう", "えーっと", "かな", "思う"],
        "images": ["think1.png", "think2.png"]
    },
    "default": {
        "keywords": [], 
        "images": [
            "normal1.png", "normal3.png", "normal5.png", 
            "normal6.png", "normal7.png", "normal8.png", "normal10.png", 
            "normal11.png", "normal13.png"
        ]
    }
}

# ==========================================
# 3. デバイス判定
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
# 4. モデルロード
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
# 5. ユーティリティ
# ==========================================

def normalize(content):
    if isinstance(content, list):
        return "".join(str(x) for x in content)
    return content or ""

def get_character_image_by_keyword(sentence):
    target_images = []
    for category, data in EMOTION_SETTINGS.items():
        if category == "default":
            continue
        if any(keyword in sentence for keyword in data["keywords"]):
            target_images = data["images"]
            break

    if not target_images:
        target_images = EMOTION_SETTINGS["default"]["images"]

    if target_images:
        valid_candidates = []
        for img_name in target_images:
            path = os.path.join(CHAR_IMAGE_DIR, img_name)
            if os.path.exists(path):
                valid_candidates.append(path)
        
        if valid_candidates:
            return random.choice(valid_candidates)
    
    return get_random_character_image()

def get_random_waiting_image():
    candidates = (
        EMOTION_SETTINGS["thinking"]["images"] + 
        EMOTION_SETTINGS["default"]["images"]
    )
    valid_paths = []
    for img_name in candidates:
        path = os.path.join(CHAR_IMAGE_DIR, img_name)
        if os.path.exists(path):
            valid_paths.append(path)
            
    if valid_paths:
        return random.choice(valid_paths)
    return get_random_character_image()

def get_random_character_image():
    if not os.path.exists(CHAR_IMAGE_DIR):
        return None
    images = [
        os.path.join(CHAR_IMAGE_DIR, f)
        for f in os.listdir(CHAR_IMAGE_DIR)
        if f.lower().endswith(".png")
    ]
    return random.choice(images) if images else None

def split_sentences(text):
    pattern = r'([。！？\n]+)'
    parts = re.split(pattern, text)
    sentences = []
    current = ""
    for part in parts:
        current += part
        if re.match(pattern, part):
            sentences.append(current)
            current = ""
    if current:
        sentences.append(current)
    return sentences

# ==========================================
# 6. ストリーミング生成
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

    buffer = ""
    displayed_text = ""
    
    for new_text in streamer:
        buffer += new_text
        clean_buffer = buffer.replace("<NL>", "\n")
        
        sentences = split_sentences(clean_buffer)
        
        if len(sentences) > 1 or (len(sentences) == 1 and any(sentences[0].endswith(end) for end in ["。", "！", "？", "\n"])):
            for i, sentence in enumerate(sentences[:-1]):
                if displayed_text and not displayed_text.endswith("\n"):
                    displayed_text += "\n"
                displayed_text += sentence
                yield displayed_text, sentence
                time.sleep(SENTENCE_PAUSE_DURATION)
            
            if len(sentences) > 0 and any(sentences[-1].endswith(end) for end in ["。", "！", "？", "\n"]):
                if displayed_text and not displayed_text.endswith("\n"):
                    displayed_text += "\n"
                displayed_text += sentences[-1]
                yield displayed_text, sentences[-1]
                time.sleep(SENTENCE_PAUSE_DURATION)
                buffer = ""
            else:
                buffer = sentences[-1] if sentences else ""
    
    if buffer:
        clean_buffer = buffer.replace("<NL>", "\n")
        if displayed_text and not displayed_text.endswith("\n"):
            displayed_text += "\n"
        displayed_text += clean_buffer
        yield displayed_text, clean_buffer

# ==========================================
# 7. Gradio 応答関数
# ==========================================

def respond(message, history):
    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]

    waiting_image = get_random_waiting_image()
    yield new_history, waiting_image

    generator = stream_generate(message, history)
    
    for text, completed_sentence in generator:
        new_history[-1]["content"] = text
        
        if completed_sentence:
            current_image = get_character_image_by_keyword(completed_sentence)
            yield new_history, current_image
        else:
            pass 

# ==========================================
# 8. Gradio UI
# ==========================================

with gr.Blocks(css=CUSTOM_CSS) as demo:
    gr.Markdown("## 🎭 AITuber Nemu (Proto) - Sentence-based Image Switching")
    gr.Markdown("*各文ごとにキーワードを判定して、キャラクター表情を切り替えます*")

    with gr.Row():
        chatbot = gr.Chatbot(
            height=700,
        )
        character = gr.Image(
            label="Character",
            type="filepath",
            height=700,
            value=DEFAULT_IMAGE_PATH if os.path.exists(DEFAULT_IMAGE_PATH) else None,
            elem_id="character-view",   
            show_label=False,           
            interactive=False,          # 編集不可
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