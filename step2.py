
from transformers import AutoTokenizer
#-------------------------------------------------------------------------------------
# Step 2: ベースモデルの設定確認とファインチューニングへの準備
#-------------------------------------------------------------------------------------
# モデル情報
model_name = "ToPo-ToPo/rinna-japanese-gpt-neox-3.6b"

# トークナイザーの準備
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# トークナイザーのスペシャルトークンの確認
print(tokenizer.special_tokens_map)
print("bos_token :", tokenizer.bos_token, ",", tokenizer.bos_token_id)
print("eos_token :", tokenizer.eos_token, ",", tokenizer.eos_token_id)
print("unk_token :", tokenizer.unk_token, ",", tokenizer.unk_token_id)
print("pad_token :", tokenizer.pad_token, ",", tokenizer.pad_token_id)

# トークナイズ関数の定義
CUTOFF_LEN = 256  # コンテキスト長

# トークナイズ関数の定義(Rinna用)
def tokenize(prompt, tokenizer):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
    )
    return {
        "input_ids": result["input_ids"],
        "attention_mask": result["attention_mask"],
    }

# トークナイズ関数の動作確認
result = tokenize("hi there", tokenizer)
print(result)