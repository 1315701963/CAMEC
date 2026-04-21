llamafactory-cli export \
    --model_name_or_path ./model/qwen3-8b \
    --adapter_name_or_path ./model/lora/sft_output \
    --template qwen3 \
    --finetuning_type lora \
    --trust_remote_code True \
    \
    --export_dir ./model/lora_export_stage4 \
    --export_size 2 \
    --export_device auto \
    --export_legacy_format False
