# Chinese-Medical-LLM
A multi-agent Chinese medical large language model system combining retrieval, reasoning, and knowledge graph.

# Install dependencies
pip install -r requirements.txt

# Run Fine-tuning
Each stage has its own training and export script located in finetune/parameters.

▶️ Stage 1
```bash
bash sft_stage1_train.sh
bash export_stage1.sh
```

▶️ Stage 2
```bash
bash sft_stage2_train.sh
bash export_stage2.sh
```

▶️ Stage 3
```bash
bash sft_stage3_train.sh
bash export_stage3.sh
```

▶️ Stage 4
```bash
bash sft_stage4_train.sh
bash export_stage4.sh
```

After each export, the LoRA adapters are merged into the base model for the next stage.
