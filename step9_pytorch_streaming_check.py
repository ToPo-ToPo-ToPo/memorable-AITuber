
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

#====================================================================
# 設定エリア
#====================================================================
base_model_name = "ToPo-ToPo/ai-character-suuchi-kai-3.6b"

# システムプロンプト（キャラ設定）
CHARACTER_SYSTEM_PROMPT = """
あなたは「解析カイ」という名前の新人アシスタントです。女の子です。
親しみやすいタメ口で会話します。
"""

#====================================================================
# モデルとトークナイザーの準備
#====================================================================
print("モデルを読み込んでいます...")

# 1. トークナイザーの読み込み
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. モデルの読み込み
# Mac(Apple Silicon)向けに最適化: device_map="mps", torch_dtype=torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="mps",
    #torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

# 推論モードに固定
model.eval()

print("準備完了。会話を開始します。")
print("-" * 60)

#====================================================================
# 推論用関数（ストリーミング対応）
#====================================================================
@torch.inference_mode()
def generate_response_stream(instruction, input_context=None):
    """
    質問を受け取り、回答をリアルタイムでコンソールに表示します。
    """
    # 1. プロンプトの作成
    # Rinna形式: Instruction / Input / Response
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
    
    # 2. トークナイズ
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        add_special_tokens=False
    ).to(model.device)
    
    # 3. ストリーマーの設定（ここがポイント）
    # skip_prompt=True: プロンプト（指示文）は再表示せず、生成された回答だけを表示
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # 4. 生成実行
    # model.generate の結果は streamer を通じて標準出力に直接流れます
    _ = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.85,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer  # ストリーマーを指定
    )
    
    # TextStreamerは自動で改行しないことがあるため、最後に改行を入れる
    print() 

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
    
    # ウォームアップ（初回のみロードやコンパイルで時間がかかるため）
    print("ウォームアップ中（最初の1回は少し待ちます）...", end="", flush=True)
    # ウォームアップ時はストリーミングせず、内部で空回しする
    dummy_input = tokenizer("test", return_tensors="pt").to(model.device)
    model.generate(**dummy_input, max_new_tokens=1)
    print("完了！\n")
    
    # 本番ループ
    for i, q in enumerate(questions, 1):
        print(f"Q{i}: {q}")
        print("A: ", end="", flush=True) # "A: "と表示して待機
        
        # ここで文字が流れます
        generate_response_stream(instruction=q, input_context=CHARACTER_SYSTEM_PROMPT)
        
        print("-" * 60)