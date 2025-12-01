
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

tokenizer = AutoTokenizer.from_pretrained("line-corporation/japanese-large-lm-3.6b-instruction-sft", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("line-corporation/japanese-large-lm-3.6b-instruction-sft")
 
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
 
input_text = """四国の県名を全て列挙してください。"""
text = generator(
    f"ユーザー: {input_text}\nシステム: ",
    max_length = 256,
    do_sample = True,
    temperature = 0.7,
    top_p = 0.9,
    top_k = 0,
    repetition_penalty = 1.1,
    num_beams = 1,
    pad_token_id = tokenizer.pad_token_id,
    num_return_sequences = 1,
)
print(text)
# [{'generated_text': 'ユーザー: 四国の県名を全て列挙してください。\nシステム:  高知県、徳島県、香川県、愛媛県'}]
