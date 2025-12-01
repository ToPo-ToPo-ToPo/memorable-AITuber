
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed

# macOSの場合
model = AutoModelForCausalLM.from_pretrained("line-corporation/japanese-large-lm-3.6b")
tokenizer = AutoTokenizer.from_pretrained("line-corporation/japanese-large-lm-3.6b", use_fast=False)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device="cpu")

# オリジナル
#model = AutoModelForCausalLM.from_pretrained("line-corporation/japanese-large-lm-3.6b", torch_dtype=torch.float16)
#tokenizer = AutoTokenizer.from_pretrained("line-corporation/japanese-large-lm-3.6b", use_fast=False)
#generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
set_seed(101)

text = generator(
   "おはようございます、今日の天気は",
   max_length=30,
   do_sample=True,
   pad_token_id=tokenizer.pad_token_id,
   num_return_sequences=1,
)
 
for t in text:
   print(t)
