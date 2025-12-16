
#-------------------------------------------------------------------------------------
# Step 1: ベースモデルをとにかく動かす。
# Huggingfaceの公式リポジトリがなくなっていたので、自身のcacheにたまたま残っていたモデルを使用
#
# モデルロード後初めての回答
# Q:まどか☆マギカでは誰が一番かわいい? 
# A:さやかちゃん! ●●●はやっぱり、おっぱいが大きい子! ●●●は大きいのに、やわらかいのです。 
# ●●●は、おっぱいが大きいだけで、なんでも許してしまいます。 ●●●の魅力は、大きいのに、や
# 別の解答例
# Q:まどか☆マギカでは誰が一番かわいい? 
# A:それはもう、マミさんに決まってるじゃないですかー!! 
# ってお前はキュゥべえか!(注:キュゥべえはマスコットキャラ的なもので、決してエロいお兄ちゃんではありません)
# Q:まどか☆マギカってなんでこんなに人気あるの? A
# 別の回答例
# Q:まどか☆マギカでは誰が一番かわいい? 
# A:アルティメットまどかです B:え? もしかして一番好きなの? 
# C:もちろん一番かわいいのはさやかちゃんです 
# D:えっと・・・、私だったらマミさんかな。でも一番はキュウべぇだよ! まどマギの話で、こういうこと言うと
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