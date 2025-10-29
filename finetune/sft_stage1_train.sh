llamafactory-cli train \
    --stage sft \
    --do_train True \
    --finetuning_type lora \
    --lora_rank 16 \
    --lora_target "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
    --lora_dropout 0.05 \
    \
    --model_name_or_path /home/zf1/WuYukang/AIstorian/model/Qwen3-8B \
    --trust_remote_code True \
    --image_max_pixels 262144 \
    --video_max_pixels 16384 \
    \
    --dataset ApolloMoEDataset_sample \
    --dataset_dir ./dataset \
    --template qwen3 \
    --cutoff_len 1024 \
    --val_size 0.02 \
    --overwrite_cache True \
    --preprocessing_num_workers 8 \
    --dataloader_num_workers 2 \
    \
    --output_dir /home/zf1/WuYukang/AIstorian/model/lora/sft_stage1 \
    --logging_steps 10 \
    --save_steps 1000 \
    --plot_loss True \
    --save_only_model False \
    --report_to none \
    --overwrite_output_dir False \
    --resume_from_checkpoint True \
    \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 1.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --bf16 True \
    --gradient_checkpointing True \
    --torch_compile True \
    --optim adamw_torch \
    --max_grad_norm 1.0 \
    --ddp_timeout 180000000 \
    \
    --per_device_eval_batch_size 2 \
    --eval_strategy steps \
    --eval_steps 20000