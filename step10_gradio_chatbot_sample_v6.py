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

MODEL_NAME = "rinna/japanese-gpt-neox-3.6b-instruction-sft-v2"
CHAR_IMAGE_DIR = "assets/characters"

SYSTEM_PROMPT = (
    "あなたはAITuberの『根夢（ねむ）』です。"
    "明るく親しみやすく、短めに会話してください。"
    "難しい言葉は使わず、自然な日本語で話します。"
)

MAX_NEW_TOKENS = 256

# 文が完成してから次の文開始までの待機時間（秒）
SENTENCE_PAUSE_DURATION = 2.0

# ==========================================
# 2. キーワードと画像の対応
# ==========================================

# キーワードごとに表示する画像ファイル名を定義
KEYWORD_IMAGE_MAP = {
    "嬉しい": "happy.png",
    "楽しい": "happy.png",
    "わーい": "happy.png",
    "やった": "happy.png",
    "最高": "happy.png",
    "悲しい": "sad.png",
    "つらい": "sad.png",
    "残念": "sad.png",
    "困": "sad.png",
    "怒": "angry.png",
    "ムカ": "angry.png",
    "イライラ": "angry.png",
    "驚": "surprised.png",
    "びっくり": "surprised.png",
    "え！": "surprised.png",
    "へー": "surprised.png",
    "考え": "thinking.png",
    "うーん": "thinking.png",
    "どうしよう": "thinking.png",
    # デフォルト(キーワードマッチしない場合)
    "default": "normal.png",
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
    """Gradio 6.x の content 正規化"""
    if isinstance(content, list):
        return "".join(str(x) for x in content)
    return content or ""

def get_character_image_by_keyword(sentence):
    """
    文章からキーワードを検出して対応する画像パスを返す
    """
    # キーワードをチェック
    for keyword, image_name in KEYWORD_IMAGE_MAP.items():
        if keyword == "default":
            continue
        if keyword in sentence:
            image_path = os.path.join(CHAR_IMAGE_DIR, image_name)
            if os.path.exists(image_path):
                return image_path
    
    # マッチしない場合はデフォルト画像
    default_path = os.path.join(CHAR_IMAGE_DIR, KEYWORD_IMAGE_MAP["default"])
    if os.path.exists(default_path):
        return default_path
    
    # デフォルト画像もない場合はランダム
    return get_random_character_image()

def get_random_character_image():
    """ランダムにキャラクター画像を取得"""
    images = [
        os.path.join(CHAR_IMAGE_DIR, f)
        for f in os.listdir(CHAR_IMAGE_DIR)
        if f.lower().endswith(".png")
    ]
    return random.choice(images) if images else None

def split_sentences(text):
    """
    テキストを文単位で分割する
    。、！、？で区切る
    """
    # 文末記号で分割
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
        
        # 文が完成したかチェック
        sentences = split_sentences(clean_buffer)
        
        # 完成した文があるか確認
        if len(sentences) > 1 or (len(sentences) == 1 and any(sentences[0].endswith(end) for end in ["。", "！", "？", "\n"])):
            # 完成した文を取得
            for i, sentence in enumerate(sentences[:-1]):
                displayed_text += sentence
                # 完成した文を表示し、その文に基づいて画像を選択
                yield displayed_text, sentence
                # 文の終わりで少し待機
                time.sleep(SENTENCE_PAUSE_DURATION)
            
            # 最後の要素が完成した文の場合
            if len(sentences) > 0 and any(sentences[-1].endswith(end) for end in ["。", "！", "？", "\n"]):
                displayed_text += sentences[-1]
                yield displayed_text, sentences[-1]
                time.sleep(SENTENCE_PAUSE_DURATION)
                buffer = ""
            else:
                # 未完成の部分はバッファに残す
                buffer = sentences[-1] if sentences else ""
    
    # 最後に残ったテキストがあれば表示
    if buffer:
        clean_buffer = buffer.replace("<NL>", "\n")
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

    generator = stream_generate(message, history)
    current_image = get_random_character_image()

    for text, completed_sentence in generator:
        new_history[-1]["content"] = text
        
        # 完成した文に基づいて画像を更新
        if completed_sentence:
            current_image = get_character_image_by_keyword(completed_sentence)
        
        yield new_history, current_image

# ==========================================
# 8. Gradio UI
# ==========================================

with gr.Blocks() as demo:
    gr.Markdown("## 🎭 AITuber Nemu (Proto) - Sentence-based Image Switching")
    gr.Markdown("*各文ごとにキーワードを判定して、キャラクター表情を切り替えます*")

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