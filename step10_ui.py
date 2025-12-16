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
# 設定エリア
# ============================================================
BASE_MODEL_NAME = "ToPo-ToPo/ai-character-suuchi-kai-3.6b"

CHARACTER_SYSTEM_PROMPT = """
あなたは「解析カイ」という名前の新人アシスタントです。女の子です。
親しみやすいタメ口で会話します。
"""

STREAM_DELAY = 0.05
MAX_NEW_TOKENS = 256

# ============================================================
# モデル準備
# ============================================================
print(f"Gradio version: {gr.__version__}")
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
    #torch_dtype=torch.bfloat16, # 高速化
    low_cpu_mem_usage=True
)

model.eval()
print("準備完了")

# ============================================================
# ★重要：データクリーニング関数
# ============================================================
def get_clean_text(msg):
    """
    Gradio 6.x の複雑なデータ構造から、純粋なテキストだけを抽出する関数
    入力が辞書でもオブジェクトでも、中身がリストでも文字列でも対応します。
    """
    # 1. まずメッセージオブジェクトから content と role を取り出す
    if isinstance(msg, dict):
        content = msg.get('content', '')
    else:
        # ChatMessageオブジェクトの場合
        content = getattr(msg, 'content', '')

    # 2. content がリスト形式（[{'text': '...', 'type': 'text'}]）の場合の処理
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get('type') == 'text':
                text_parts.append(item.get('text', ''))
        return "".join(text_parts)
    
    # 3. 単純な文字列の場合
    return str(content)

def get_role(msg):
    """ロールを取り出す"""
    if isinstance(msg, dict):
        return msg.get('role', '')
    return getattr(msg, 'role', '')

# ============================================================
# 推論ロジック
# ============================================================

def add_user_message(user_message, history):
    if not user_message:
        return "", history
    if history is None:
        history = []
    
    # UI表示用には ChatMessage をそのまま使う
    history.append(gr.ChatMessage(role="user", content=user_message))
    return "", history

@torch.inference_mode()
def bot_stream(history):
    if not history:
        yield history
        return

    # ---------------------------------------------------------
    # プロンプト構築（クリーニング関数を通す！）
    # ---------------------------------------------------------
    prompt = ""
    
    # 1. 過去の履歴（直前以外）
    for msg in history[:-1]:
        role = get_role(msg)
        text = get_clean_text(msg) # ★ここで綺麗なテキストにする
        
        if role == "user":
            prompt += f"### 指示:\n{text}\n\n"
        elif role == "assistant":
            prompt += f"### 回答:\n{text}\n\n"
    
    # 2. 今回のターン
    last_msg = history[-1]
    current_user_text = get_clean_text(last_msg) # ★ここも綺麗にする
    
    prompt += (
        f"### 指示:\n{current_user_text}\n\n"
        f"### 入力:\n{CHARACTER_SYSTEM_PROMPT.strip()}\n\n"
        f"### 回答:\n"
    )

    # デバッグ用：今度こそ綺麗なテキストが出ているか確認
    print("\n--- 生成プロンプト確認 ---\n" + prompt + "\n------------------------")

    # ---------------------------------------------------------
    # 生成処理
    # ---------------------------------------------------------
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

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 履歴にAI枠を追加
    ai_msg = gr.ChatMessage(role="assistant", content="")
    history.append(ai_msg)

    generated_text = ""
    for token in streamer:
        generated_text += token
        clean_text = generated_text.replace("<NL>", "\n")
        
        # UI更新（オブジェクトのcontentを書き換え）
        # ※オブジェクトなら直接書き換えてもリスト化されにくいが、念のため
        if isinstance(history[-1], dict):
             history[-1]['content'] = clean_text
        else:
             history[-1].content = clean_text
        
        yield history
        
        if clean_text.endswith(("。", "！", "？", "\n")):
            time.sleep(STREAM_DELAY * 3)
        else:
            time.sleep(STREAM_DELAY)

# ============================================================
# UI構築
# ============================================================
with gr.Blocks(title="解析カイ Chatbot") as demo:
    gr.Markdown(f"## 🤖 解析カイ (Gradio v{gr.__version__} Fixed)")
    
    chatbot = gr.Chatbot(height=500)

    with gr.Row():
        textbox = gr.Textbox(
            placeholder="メッセージを入力してEnter...",
            show_label=False,
            scale=9
        )
        clear_btn = gr.Button("クリア", scale=1)

    textbox.submit(
        add_user_message, [textbox, chatbot], [textbox, chatbot], queue=False
    ).then(
        bot_stream, [chatbot], [chatbot]
    )

    clear_btn.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.queue().launch()