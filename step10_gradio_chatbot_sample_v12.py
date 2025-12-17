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

# モデル設定 (Code Bより)
MODEL_NAME = "ToPo-ToPo/ai-character-suuchi-kai-3.6b-v2"
CHAR_IMAGE_DIR = "assets/characters_v1"

# システムプロンプト (Code Bより)
SYSTEM_PROMPT = """
あなたは「数値カイ」という名前の新人アシスタントです。女の子です。
親しみやすいタメ口で会話します。
""".strip()

# 生成パラメータ (Code Bより調整)
MAX_NEW_TOKENS = 1024
MEMORY_TURNS = 2       # 記憶する過去の会話往復数
SENTENCE_PAUSE_DURATION = 1.0 # 画像切り替えのためのウェイト

DEFAULT_IMAGE_PATH = os.path.join(CHAR_IMAGE_DIR, "normal1.png")

# CSS定義 (Code Aを維持)
CUSTOM_CSS = """
/* ▼ メインの枠 */
#character-view {
    height: 700px !important;
    min-height: 700px !important;
    
    /* 【修正】枠線を表示する設定に変更 */
    border: 1px solid #cbd5e1 !important;  /* グレーの枠線を追加 */
    
    background-color: transparent !important;
    padding: 0 !important;
}

/* ▼ 画像表示エリアの高さを確保 */
#character-view .wrap,
#character-view .image-container,
#character-view .image-frame {
    height: 100% !important;
    min-height: 100% !important;
    display: flex !important;
    justify-content: center;
    align-items: flex-end; /* 下揃えの補助 */
    border: none !important; /* 内部の二重線を防止 */
}

/* ▼ 画像本体の設定 */
#character-view img {
    width: 100% !important;
    height: 100% !important;
    object-fit: contain !important;
    transform: scale(1.3);         /* 1.3倍ズーム */
    transform-origin: center 5%;   /* 上側基準 */
    display: block !important;
}

/* ▼ 右上のダウンロードボタンや拡大ボタンだけを消す設定 */
#character-view .download,
#character-view .icon-button,
#character-view [aria-label="Download"],
#character-view [aria-label="Maximize"] {
    display: none !important;
    visibility: hidden !important;
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

print(f"モデル読み込み中: {MODEL_NAME}")
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

def clean_response(text):
    """
    Code B由来: 生成されたテキストが文の途中で終わっている場合、
    最後の句読点までカットして整える関数。
    """
    ends = ["。", "！", "？", "\n", "!", "?", "."]
    stripped_text = text.strip()
    
    if any(stripped_text.endswith(e) for e in ends):
        return stripped_text

    for i in range(len(stripped_text) - 1, -1, -1):
        if stripped_text[i] in ends:
            return stripped_text[:i+1]
    
    return stripped_text

# ==========================================
# 6. プロンプト構築 & ストリーミング生成
# ==========================================

def build_prompt(current_message, history):
    """
    Code B由来のロジックをGradioの履歴形式に適用
    """
    prompt = ""
    
    # 1. Gradioの履歴(list of dicts)を (user, assistant) のペアに変換
    pairs = []
    current_user_text = None
    
    for msg in history:
        role = msg['role']
        content = normalize(msg['content'])
        
        if role == 'user':
            current_user_text = content
        elif role == 'assistant' and current_user_text is not None:
            pairs.append((current_user_text, content))
            current_user_text = None
    
    # 2. 直近 MEMORY_TURNS 分だけ取得 (Code Bロジック)
    if MEMORY_TURNS > 0:
        recent_pairs = pairs[-MEMORY_TURNS:]
    else:
        recent_pairs = []

    for user_text, bot_text in recent_pairs:
        safe_user = user_text.replace("\n", "<NL>")
        safe_bot = bot_text.replace("\n", "<NL>")
        prompt += f"ユーザー: {safe_user}<NL>システム: {safe_bot}<NL>"

    # 3. 今回の入力を追加 (システムプロンプトをここに埋め込む: Code Bロジック)
    safe_current = current_message.strip().replace("\n", "<NL>")
    safe_system = SYSTEM_PROMPT.replace("\n", "<NL>")
    
    prompt += f"ユーザー: {safe_system}<NL>{safe_current}<NL>システム: "
    
    return prompt

def stream_generate(message, history):
    # ---------------------------------------------------------
    # プロンプト作成 (Code Bのロジックを使用)
    # ---------------------------------------------------------
    prompt = build_prompt(message, history)

    # ---------------------------------------------------------
    # 生成処理
    # ---------------------------------------------------------
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
        temperature=0.85,
        top_k=50,
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
        # Code Bロジック: <NL>を改行に戻す
        clean_buffer = buffer.replace("<NL>", "\n")
        
        # 暴走対策
        if "ユーザー:" in clean_buffer:
            clean_buffer = clean_buffer.split("ユーザー:")[0]
            if displayed_text and not displayed_text.endswith("\n"):
                displayed_text += "\n"
            displayed_text += clean_buffer
            yield displayed_text, clean_buffer
            return

        # Code Aロジック: 文単位で分割して画像切り替えタイミングを作る
        sentences = split_sentences(clean_buffer)
        
        if len(sentences) > 1 or (len(sentences) == 1 and any(sentences[0].endswith(end) for end in ["。", "！", "？", "\n"])):
            for i, sentence in enumerate(sentences[:-1]):
                if displayed_text and not displayed_text.endswith("\n") and not sentence.startswith("\n"):
                    displayed_text += "\n"
                displayed_text += sentence
                yield displayed_text, sentence
                time.sleep(SENTENCE_PAUSE_DURATION)
            
            last_part = sentences[-1]
            if len(sentences) > 0 and any(last_part.endswith(end) for end in ["。", "！", "？", "\n"]):
                if displayed_text and not displayed_text.endswith("\n") and not last_part.startswith("\n"):
                    displayed_text += "\n"
                displayed_text += last_part
                yield displayed_text, last_part
                time.sleep(SENTENCE_PAUSE_DURATION)
                buffer = ""
            else:
                buffer = last_part if sentences else ""
    
    if buffer:
        clean_buffer = buffer.replace("<NL>", "\n")
        if "ユーザー:" in clean_buffer:
            clean_buffer = clean_buffer.split("ユーザー:")[0]
            
        if displayed_text and not displayed_text.endswith("\n"):
            displayed_text += "\n"
        displayed_text += clean_buffer
        yield displayed_text, clean_buffer

# ==========================================
# 7. Gradio 応答関数
# ==========================================

def respond(message, history):
    # Gradio推奨のリスト形式で履歴更新
    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": ""},
    ]

    waiting_image = get_random_waiting_image()
    yield new_history, waiting_image

    # 生成開始 (historyは直前の状態を渡して build_prompt 内で処理)
    generator = stream_generate(message, history)
    
    final_text = ""
    for text, completed_sentence in generator:
        final_text = text
        new_history[-1]["content"] = text
        
        if completed_sentence:
            current_image = get_character_image_by_keyword(completed_sentence)
            yield new_history, current_image
        else:
            yield new_history, gr.Skip() # 画像変更なし

    # Code Bロジック: 文末カット処理 (clean_response) を最後に適用
    cleaned_text = clean_response(final_text)
    if cleaned_text != final_text:
        new_history[-1]["content"] = cleaned_text
        yield new_history, gr.Skip()

# ==========================================
# 8. Gradio UI
# ==========================================

with gr.Blocks(css=CUSTOM_CSS) as demo:
    gr.Markdown("## 🎭 解析カイ (Character AI) - Image Switching Demo")
    gr.Markdown("*文脈を読んで表情を変えながら会話します*")

    with gr.Row():
        chatbot = gr.Chatbot(
            height=700,
            show_label=False  # チャットのラベルを非表示
            # type="messages" は指定しない（Code A準拠）
        )
        character = gr.Image(
            label="Character",
            type="filepath",
            height=700,
            value=DEFAULT_IMAGE_PATH if os.path.exists(DEFAULT_IMAGE_PATH) else None,
            elem_id="character-view",   
            show_label=False,           # 画像のラベルを非表示
            interactive=False,
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