import torch
import time
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

#====================================================================
# 設定
#====================================================================
base_model_name = "ToPo-ToPo/ai-character-suuchi-kai-3.6b"

CHARACTER_SYSTEM_PROMPT = """
あなたは「解析カイ」という名前の新人アシスタントです。女の子です。
親しみやすいタメ口で会話します。
"""

#====================================================================
# ★滑らか表示用のクラスを作成（ここが改造ポイント）
#====================================================================
class SmoothStreamer(TextStreamer):
    def __init__(self, tokenizer, skip_prompt=True, delay=0.02, **decode_kwargs):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.delay = delay  # 1文字ごとの待機時間（秒）

    def on_finalized_text(self, text: str, stream_end: bool = False):
        # AIから来た「塊（トークン）」を、1文字ずつバラして表示する
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(self.delay) # ここでわざと遅らせて滑らかに見せる

#====================================================================
# モデル準備
#====================================================================
print("モデルを読み込んでいます...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="mps",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
model.eval()
print("準備完了。")
print("-" * 60)

#====================================================================
# 推論関数
#====================================================================
@torch.inference_mode()
def generate_response_smooth(instruction, input_context=None):
    if input_context:
        prompt = f"### 指示:\n{instruction}\n\n### 入力:\n{input_context}\n\n### 回答:\n"
    else:
        prompt = f"### 指示:\n{instruction}\n\n### 回答:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    
    # ★自作した滑らかストリーマーを使う
    # delay=0.03 くらいが読みやすい速度です（数字を増やすとゆっくりになります）
    streamer = SmoothStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, delay=0.03)
    
    _ = model.generate(
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
    print() # 最後に改行

#====================================================================
# 実行
#====================================================================
if __name__ == "__main__":
    questions = [
        "自己紹介をしてくれる？",
        "美味しいカレーの作り方を教えて。",
    ]
    
    # ウォームアップ
    print("ウォームアップ中...", end="", flush=True)
    dummy = tokenizer("test", return_tensors="pt").to(model.device)
    model.generate(**dummy, max_new_tokens=1)
    print("完了！\n")
    
    for q in questions:
        print(f"Q: {q}")
        print("A: ", end="", flush=True)
        generate_response_smooth(instruction=q, input_context=CHARACTER_SYSTEM_PROMPT)
        print("-" * 60)