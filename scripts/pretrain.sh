
# /home/hukun/anaconda3/envs/llama/bin/python \
# export PYTHONPATH=/home/hukun/work/Tmm_llama-main:$PYTHONPATH
accelerate launch  --config_file=./configs/single_gpu.yaml llatmm/train.py \
    --model_name_or_path /home/hukun/work/mol_llama/llama-2-7B-chat \
    --data_path ./data/test_train.jsonl \
    --mol_folder ./data/mol_features \
    --model_max_length 2048 \
    --is_multimodal True \
    --mol_projector True \
    --bf16 True \
    --lora_enable True \
    --output_dir ./checkpoints/pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \

    