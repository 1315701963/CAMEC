llamafactory-cli train \
    --stage sft \
    --do_train True \
    --finetuning_type lora \
    --lora_rank 16 \
    --lora_target "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --lora_dropout 0.05 \
    \
    --model_name_or_path ./model/lora_export_stage2 \
    --trust_remote_code True \
    --image_max_pixels 262144 \
    --video_max_pixels 16384 \
    \
    --dataset HuatuoGPT2_sample \
    --dataset_dir ./dataset \
    --template qwen3 \
    --cutoff_len 2048 \
    --val_size 0.02 \
    --overwrite_cache True \
    --preprocessing_num_workers 8 \
    --dataloader_num_workers 2 \
    \
    --output_dir ./model/lora/sft_stage3 \
    --logging_steps 10 \
    --save_steps 1000 \
    --plot_loss True \
    --save_only_model False \
    --report_to none \
    --overwrite_output_dir True \
    --resume_from_checkpoint None \
    \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 True \
    --ddp_timeout 180000000 \
    --gradient_checkpointing True \
    \
    --per_device_eval_batch_size 2 \
    --eval_strategy steps \
    --eval_steps 5000
