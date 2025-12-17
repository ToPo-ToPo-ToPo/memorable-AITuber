
#-------------------------------------------------------------------------------------
# Step 1: ベースモデルをとにかく動かす。
# Huggingfaceの公式リポジトリがなくなっていたので、自身のcacheにたまたま残っていたモデルを使用
#-------------------------------------------------------------------------------------

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# トークナイザーとモデルの準備
tokenizer = AutoTokenizer.from_pretrained(
    "ToPo-ToPo/rinna-japanese-gpt-neox-3.6b", 
    use_fast=False
)
model = AutoModelForCausalLM.from_pretrained(
    "ToPo-ToPo/rinna-japanese-gpt-neox-3.6b"
).to("mps")

# プロンプトの準備
prompt = "Q:まどか☆マギカでは誰が一番かわいい？\nA:"

# 推論の実行
token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
with torch.no_grad():
    output_ids = model.generate(
        token_ids.to(model.device),
        max_new_tokens=64,
        min_new_tokens=64,
        do_sample=True,
        temperature=0.8,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

# 出力
output = tokenizer.decode(output_ids.tolist()[0])
print(output)