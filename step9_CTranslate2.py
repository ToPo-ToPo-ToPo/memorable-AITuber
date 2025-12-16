
#https://note.com/__olender/n/n7913ac32c18c

import subprocess

base_model = "ToPo-ToPo/ai-character-suuchi-kai-3.6b" #変換したいモデルを指定
output_dir = "./models/ai-character-suuchi-kai-3.6b-Ctranslate2"  #CTranslate2への変換先
quantization_type = "bfloat16"

command = f"ct2-transformers-converter --model {base_model} --output_dir {output_dir}"
#command = f"ct2-transformers-converter --model {base_model} --quantization {quantization_type} --output_dir {output_dir}"
subprocess.run(command, shell=True)