
#-------------------------------------------------------------------------
# 最適化版: PyTorch推論（時間測定付き）
#-------------------------------------------------------------------------
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

#====================================================================
# 設定
#====================================================================
base_model_name = "ToPo-ToPo/ai-character-suuchi-kai-3.6b"
CHARACTER_SYSTEM_PROMPT = """
あなたは「解析カイ」という名前の新人アシスタントです。女の子です。
親しみやすいタメ口で会話します。
"""

#====================================================================
# モデルとトークナイザーの準備
#====================================================================
print("モデルを読み込んでいます...")

# 1. モデルの読み込み（最適化版）
# bfloat16を使用してメモリと速度を改善
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="mps",
    #torch_dtype=torch.bfloat16,  # float32 → bfloat16で高速化
    low_cpu_mem_usage=True,
)

# 評価モードに設定（推論のみの場合）
model.eval()

# 2. トークナイザーの読み込み
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("準備完了。\n")

#====================================================================
# 推論用関数の定義
#====================================================================
@torch.inference_mode()  # torch.no_grad() より高速
def generate_response(instruction, input_context=None):
    """
    質問(instruction)を受け取り、LLMの回答を返します。
    """
    # 1. プロンプトの作成
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
    
    input_token_len = inputs.input_ids.shape[1]
    
    # 3. 生成実行（最適化されたパラメータ）
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.85,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,  # KVキャッシュを有効化
    )
    
    # 4. デコード
    generated_tokens = outputs[0][input_token_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Rinna特有の <NL> を改行に戻す
    response = response.replace("<NL>", "\n")
    
    return response.strip()

#====================================================================
# 実行部分（時間測定付き）
#====================================================================
if __name__ == "__main__":
    questions = [
        "AITuberについて教えてください。",
        "日本で一番高い山はどこですか?",
        "美味しいカレーの作り方を教えて。",
        "まどか☆マギカでは誰が一番かわいい?"
    ]
    
    print("=" * 60)
    print("最適化版PyTorch推論モデル")
    print("=" * 60)
    
    # ウォームアップ（初回は遅いため）
    print("\nウォームアップ中...")
    _ = generate_response("テスト", CHARACTER_SYSTEM_PROMPT)
    print("ウォームアップ完了\n")
    
    # 本番実行（時間測定）
    print("-" * 60)
    times = []
    
    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] 質問: {q}")
        
        start = time.time()
        output = generate_response(instruction=q, input_context=CHARACTER_SYSTEM_PROMPT)
        elapsed = time.time() - start
        times.append(elapsed)
        
        print(f"回答: {output}")
        print(f"⏱️  処理時間: {elapsed:.2f}秒")
        print("-" * 60)
    
    # 統計情報
    print("\n" + "=" * 60)
    print("📊 統計情報")
    print("=" * 60)
    print(f"総処理時間: {sum(times):.2f}秒")
    print(f"平均処理時間: {sum(times)/len(times):.2f}秒")
    print(f"最速: {min(times):.2f}秒")
    print(f"最遅: {max(times):.2f}秒")
    print("=" * 60)