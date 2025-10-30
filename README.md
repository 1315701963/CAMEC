# Chinese-Medical-LLM
A multi-agent Chinese medical large language model system combining retrieval, reasoning, and knowledge graph.

## Install dependencies
```bash
pip install -r requirements.txt
```

## Supervised Fine-Tuning (SFT)
Each stage has its own training and export script located in finetune/, trining data located in finetune/dataset/, 

### Start training:
Run the sft_train.sh script to begin training:
```bash
bash sft_train.sh
```

### Export the model:
Run the lora_export.sh script to export the fine-tuned model:
```bash
bash lora_export.sh
```

After each export, the LoRA adapters are merged into the base model for the next stage.
