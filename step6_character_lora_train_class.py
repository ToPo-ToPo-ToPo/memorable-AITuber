
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from dataclasses import dataclass, field
from typing import List

# ==========================================
# 1. 設定エリア
# ==========================================
@dataclass
class TrainingConfig:
    # --- モデルとデータセット ---
    model_name: str = "ToPo-ToPo/rinna-japanese-gpt-neox-3.6b-lora-sft-v1"
    hf_dataset_id: str = "ToPo-ToPo/ai-characters-QA"
    hf_data_file: str = "dataset-Ai.jsonl"
    
    # --- 保存設定 ---
    peft_name: str = "models/lora-rinna-3.6b-phase2-ai"
    output_dir: str = "models/lora-rinna-3.6b-phase2-results"

    # --- 学習パラメータ ---
    cutoff_len: int = 1024        # コンテキスト長
    epochs: int = 10              # 学習周回数
    learning_rate: float = 3e-4   # 学習率
    batch_size: int = 1           # 1回に処理するデータ数 (メモリがきつい場合は小さく)
    grad_accum_steps: int = 4     # 勾配蓄積数 (実質バッチサイズ = batch_size * grad_accum)
    
    # --- LoRAパラメータ ---
    lora_r: int = 32              # LoRAランク (表現力: 8-64推奨)
    lora_alpha: int = 64          # LoRA係数 (rの2倍程度推奨)
    lora_dropout: float = 0.05
    # 学習対象モジュール (Rinna/GPT-NeoX系推奨)
    target_modules: List[str] = field(default_factory=lambda: [
        "query_key_value", 
        "dense", 
        "dense_h_to_4h", 
        "dense_4h_to_h"
    ])

    # --- システム設定 ---
    device_map: str = "mps"       # Mac: "mps", NVIDIA: "auto" or "cuda"
    use_float32: bool = True      # MPSで安定させるならTrue, CUDAならFalse(float16)推奨

# ==========================================
# 2. 学習実行クラス (ロジック部分)
# ==========================================
class LoRATrainer:
    def __init__(self, config: TrainingConfig):
        self.c = config
        self.tokenizer = None
        self.model = None

    def setup(self):
        """モデルとトークナイザーの準備"""
        print(f"モデルを読み込んでいます: {self.c.model_name}")
        
        # データ型の決定
        dtype = torch.float32 if self.c.use_float32 else torch.float16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.c.model_name,
            device_map=self.c.device_map,
            torch_dtype=dtype,
        )
        self.model.config.use_cache = False

        self.tokenizer = AutoTokenizer.from_pretrained(self.c.model_name, use_fast=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _process_data_function(self, data_point):
        """データセットのマッピング用関数 (内部メソッド)"""
        messages = data_point["messages"]
        
        system_text = ""
        user_text = ""
        assistant_text = ""

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                system_text = content
            elif role == "user":
                user_text = content
            elif role == "assistant":
                assistant_text = content

        # プロンプト構築
        full_prompt = (
            f"### 指示:\n{system_text}\n\n"
            f"### 入力:\n{user_text}\n\n"
            f"### 回答:\n{assistant_text}"
        )
        
        # Rinna仕様の改行変換
        full_prompt = full_prompt.replace('\n', '<NL>')
        full_prompt = full_prompt + self.tokenizer.eos_token

        # トークナイズ (長さ揃え + パディング)
        tokenized = self.tokenizer(
            full_prompt,
            truncation=True,
            max_length=self.c.cutoff_len,
            padding="max_length",
        )
        
        # ラベル作成 (-100でパディングを無視)
        input_ids = tokenized["input_ids"]
        labels = [
            -100 if token == self.tokenizer.pad_token_id else token 
            for token in input_ids
        ]
        tokenized["labels"] = labels
        
        return tokenized

    def prepare_dataset(self):
        """データセットの読み込みと加工"""
        print(f"データセット取得中: {self.c.hf_dataset_id} ({self.c.hf_data_file})")
        
        dataset = load_dataset(
            self.c.hf_dataset_id,
            data_files=self.c.hf_data_file,
            split="train"
        )
        
        # 検証用データの分割
        train_val = dataset.train_test_split(test_size=20, shuffle=True, seed=42)
        
        print("トークナイズ処理を実行中...")
        train_data = train_val["train"].shuffle().map(self._process_data_function)
        eval_data = train_val["test"].shuffle().map(self._process_data_function)
        
        print(f"学習データ数: {len(train_data)}")
        print(f"検証データ数: {len(eval_data)}")
        return train_data, eval_data

    def setup_lora(self):
        """LoRAの設定適用"""
        print("LoRAを設定しています...")
        lora_config = LoraConfig(
            r=self.c.lora_r,
            lora_alpha=self.c.lora_alpha,
            target_modules=self.c.target_modules,
            lora_dropout=self.c.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def train(self):
        """学習の実行"""
        if not self.model or not self.tokenizer:
            self.setup()
        
        train_data, eval_data = self.prepare_dataset()
        self.setup_lora()

        print("学習を開始します...")
        training_args = TrainingArguments(
            output_dir=self.c.output_dir,
            num_train_epochs=self.c.epochs,
            learning_rate=self.c.learning_rate,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=2,
            report_to="none",
            
            # デバイス・最適化設定
            optim="adamw_torch",
            per_device_train_batch_size=self.c.batch_size,
            gradient_accumulation_steps=self.c.grad_accum_steps,
            fp16=not self.c.use_float32, # float32を使うならfp16はFalse
        )

        trainer = Trainer(
            model=self.model,
            train_dataset=train_data,
            eval_dataset=eval_data,
            args=training_args,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )

        trainer.train()

        # 保存
        print("モデルを保存しています...")
        trainer.model.save_pretrained(self.c.peft_name)
        self.tokenizer.save_pretrained(self.c.peft_name)
        print(f"完了しました！ 保存先: {self.c.peft_name}")