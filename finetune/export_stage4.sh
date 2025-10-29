llamafactory-cli export \
    --model_name_or_path /home/zf1/WuYukang/AIstorian/model/lora_export_stage3 \
    --adapter_name_or_path "$LATEST_CKPT" \
    --template qwen3 \
    --finetuning_type lora \
    --trust_remote_code True \
    \
    --export_dir /home/zf1/WuYukang/AIstorian/model/lora_export_stage4 \
    --export_size 2 \
    --export_device auto \
    --export_legacy_format False