#-------------------------------------------------------------------------
# コンパイル版: PyTorch推論（torch.compile使用）
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
# PyTorchのバージョン確認
#====================================================================
print(f"PyTorch バージョン: {torch.__version__}")
torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
if torch_version < (2, 0):
    print("警告: torch.compileはPyTorch 2.0以降で使用できます")
    print("現在のバージョンでは通常モードで動作します")
    USE_COMPILE = False
else:
    print("torch.compileが利用可能です")
    USE_COMPILE = True

#====================================================================
# モデルとトークナイザーの準備
#====================================================================
print("\nモデルを読み込んでいます...")

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="mps",
    #torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)

model.eval()

# モデルのコンパイル（PyTorch 2.0以降）
if USE_COMPILE:
    print("モデルをコンパイルしています...")
    print("（初回実行時は時間がかかりますが、2回目以降は高速化されます）")
    
    # コンパイルモードの選択肢:
    # - "default": バランス型（推奨）
    # - "reduce-overhead": レイテンシ重視
    # - "max-autotune": 最大限の最適化（時間がかかる）
    
    model = torch.compile(model, mode="default")
    print("コンパイル完了")

tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("準備完了。\n")

#====================================================================
# 推論用関数の定義
#====================================================================
@torch.inference_mode()
def generate_response(instruction, input_context=None):
    """
    質問(instruction)を受け取り、LLMの回答を返します。
    """
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
    
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        add_special_tokens=False
    ).to(model.device)
    
    input_token_len = inputs.input_ids.shape[1]
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.85,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    
    generated_tokens = outputs[0][input_token_len:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    response = response.replace("<NL>", "\n")
    
    return response.strip()

#====================================================================
# 実行部分（速度測定付き）
#====================================================================
if __name__ == "__main__":
    questions = [
        "AITuberについて教えてください。",
        "日本で一番高い山はどこですか?",
        "美味しいカレーの作り方を教えて。",
        "まどか☆マギカでは誰が一番かわいい?"
    ]
    
    print("=" * 60)
    print("コンパイル版推論モデル")
    print("=" * 60)
    
    # ウォームアップ（コンパイル最適化のため複数回実行）
    print("\nウォームアップ中...")
    for i in range(3):
        print(f"  ウォームアップ {i+1}/3...")
        _ = generate_response("テスト", CHARACTER_SYSTEM_PROMPT)
    print("ウォームアップ完了\n")
    
    # 本番実行（速度測定）
    print("-" * 60)
    times = []
    
    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] 質問: {q}")
        
        start = time.time()
        output = generate_response(instruction=q, input_context=CHARACTER_SYSTEM_PROMPT)
        elapsed = time.time() - start
        times.append(elapsed)
        
        print(f"回答: {output}")
        print(f"処理時間: {elapsed:.2f}秒")
        print("-" * 60)
    
    # 統計情報
    print("\n" + "=" * 60)
    print("統計情報")
    print("=" * 60)
    print(f"総処理時間: {sum(times):.2f}秒")
    print(f"平均処理時間: {sum(times)/len(times):.2f}秒")
    print(f"最速: {min(times):.2f}秒")
    print(f"最遅: {max(times):.2f}秒")
    
    if USE_COMPILE:
        print("\n💡 ヒント: 初回実行は遅いですが、2回目以降は高速化されます")
    else:
        print("\n💡 ヒント: PyTorch 2.0以降にアップグレードすると高速化できます")
        print("   pip install --upgrade torch")