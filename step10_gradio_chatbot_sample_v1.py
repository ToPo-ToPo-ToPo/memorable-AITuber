
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# 1. 設定・モデル読み込み
# ==========================================

# ★ここを後で学習済みモデルのパスに書き換えるだけでOKです
MODEL_NAME = "rinna/japanese-gpt-neox-3.6b-instruction-sft-v2"

print(f"モデルをロードしています: {MODEL_NAME}")

# デバイス判定 (Mac/MPS対応)
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
elif torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float32 # Macはfp32が安定
else:
    device = "cpu"
    dtype = torch.float32

print(f"使用デバイス: {device}")

# トークナイザーとモデルの準備
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map=device,
    torch_dtype=dtype,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("準備完了。GUIを起動します...")

# ==========================================
# 2. 推論ロジック (Rinnaフォーマット)
# ==========================================
def generate_response(message, history):
    prompt = ""

    def normalize(content):
        if isinstance(content, list):
            return "".join(str(x) for x in content)
        return content

    paired_history = []
    current_user = None

    for item in history:
        role = item.get("role")
        content = normalize(item.get("content", ""))

        if role == "user":
            current_user = content
        elif role == "assistant" and current_user is not None:
            paired_history.append((current_user, content))
            current_user = None

    # 直近3往復のみ
    for user_turn, bot_turn in paired_history[-3:]:
        user_turn = user_turn.replace("\n", "<NL>")
        bot_turn = bot_turn.replace("\n", "<NL>")
        prompt += f"ユーザー: {user_turn}<NL>システム: {bot_turn}<NL>"

    # 今回の入力
    message = normalize(message).replace("\n", "<NL>")
    prompt += f"ユーザー: {message}<NL>システム: "

    token_ids = tokenizer.encode(
        prompt,
        add_special_tokens=False,
        return_tensors="pt"
    )

    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    output = tokenizer.decode(
        output_ids[0][token_ids.size(1):],
        skip_special_tokens=True
    )

    return output.replace("<NL>", "\n").strip()

# ==========================================
# 3. Gradio画面定義
# ==========================================
# ChatInterfaceを使うと、履歴管理などを自動でやってくれます
demo = gr.ChatInterface(
    fn=generate_response,
    title="AITuber Nemu (Proto)",
    description="Rinna 3.6B モデルとの会話テストです。",
    examples=[
        "自己紹介してください",
        "AIについて教えて",
        "今日の晩御飯なにがいい？"
    ],
)

if __name__ == "__main__":
    demo.launch()