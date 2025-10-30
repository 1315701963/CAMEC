llamafactory-cli export \
    --model_name_or_path ./model/lora_export_stage3 \
    --adapter_name_or_path ./model/lora/sft_stage4 \
    --template qwen3 \
    --finetuning_type lora \
    --trust_remote_code True \
    \
    --export_dir ./model/export_stage4 \
    --export_size 2 \
    --export_device auto \
    --export_legacy_format False
