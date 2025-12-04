

from step6 import TrainingConfig, LoRATrainer
# =========================================================================
# 3. 実行エントリーポイント
# =========================================================================
if __name__ == "__main__":

    # 設定のインスタンス化
    config = TrainingConfig(
        model_name="ToPo-ToPo/rinna-japanese-gpt-neox-3.6b-lora-sft-v1",
        hf_dataset_id="ToPo-ToPo/ai-characters-QA",
        hf_data_file="dataset-aituber-v3.jsonl",
        peft_name="models/rinna-japanese-gpt-neox-3.6b-lora-aituber-v3",
        epochs=10,
        lora_r=32,
        lora_alpha=64
    )
    
    # トレーナーの作成と実行
    trainer = LoRATrainer(config)
    trainer.train()