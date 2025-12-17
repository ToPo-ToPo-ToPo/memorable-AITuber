
#-------------------------------------------------------------------------
# Step5: 学習済みのLoRAアダプターをマージして1つのモデルを作る
#-------------------------------------------------------------------------

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
#====================================================================
# 設定
#====================================================================
# 1. 元となるベースモデル
base_model_name = "ToPo-ToPo/rinna-japanese-gpt-neox-3.6b"

# 2. 学習したLoRAアダプターのパス
peft_name = "models/lora-rinna-3.6b-v2"

# 3. マージしたモデルの保存先（わかりやすい名前をつける）
output_dir = "models/rinna-japanese-gpt-neox-3.6b-lora-sft-v2"

#====================================================================
# モデルの読み込みと結合
#====================================================================
print(f"ベースモデル ({base_model_name}) を読み込んでいます...")

# ベースモデルの読み込み
# マージ作業はメモリを消費しますが、256GBあれば問題ありません
# 精度のために float32 (または float16) を維持します
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="mps",          # MacのGPUを使用
    torch_dtype=torch.float32, # フル精度
    return_dict=True
)

# トークナイザーの読み込み
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

print(f"LoRAアダプター ({peft_name}) を結合しています...")

# ベースモデルにLoRAアダプターを適用
model = PeftModel.from_pretrained(
    base_model, 
    peft_name,
    device_map="mps",
    torch_dtype=torch.float32
)

#====================================================================
# マージ実行
#====================================================================
print("LoRAの重みをベースモデルに統合（マージ）しています...")

# merge_and_unload(): LoRAの重みを計算してベースモデルの重みに足し合わせ、
# LoRA層を削除して「普通のモデル」の状態に戻します。
merged_model = model.merge_and_unload()

#====================================================================
# 保存
#====================================================================
print(f"マージ済みモデルを保存しています... 出力先: {output_dir}")

# 保存先ディレクトリがなければ作成
os.makedirs(output_dir, exist_ok=True)

# モデルとトークナイザーを保存
merged_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("="*50)
print("完了しました！")
print(f"次の学習では、 model_name = '{output_dir}' と指定してください。")
print("="*50)