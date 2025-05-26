#!/bin/bash
BATCH_SIZE=4
GRADIENT_ACC=1
PREC=fp16

export WANDB_NAME="tinysd-b${BATCH_SIZE}-g${GRADIENT_ACC}-${PREC}"
#export WANDB_DISABLE_SERVICE=true
export CUDA_VISIBLE_DEVICES="0"

DATASET_PATH="data/ffhq_wild_files"


DATASET_NAME="ffhq"
FAMILY=segmind
MODEL=tiny-sd
IMAGE_ENCODER=openai/clip-vit-large-patch14

accelerate launch \
    --mixed_precision=${PREC} \
    -m fastcomposer.train \
    --mixed_precision=${PREC} \
    --pretrained_model_name_or_path ${FAMILY}/${MODEL} \
    --dataset_name ${DATASET_PATH} \
    --logging_dir logs/${MODEL}/${DATASET_NAME}/${WANDB_NAME} \
    --output_dir models/${MODEL}/${DATASET_NAME}/${WANDB_NAME} \
    --max_train_steps 400000 \
    --num_train_epochs 400000 \
    --train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACC \
    --learning_rate 1e-5 \
    --unet_lr_scale 1.0 \
    --checkpointing_steps 1000 \
    --allow_tf32 \
    --keep_only_last_checkpoint \
    --keep_interval 10000 \
    --seed 42 \
    --image_encoder_type clip \
    --image_encoder_name_or_path ${IMAGE_ENCODER} \
    --num_image_tokens 1 \
    --max_num_objects 4 \
    --train_resolution 512 \
    --object_resolution 224 \
    --text_image_linking postfuse \
    --object_appear_prob 0.9 \
    --uncondition_prob 0.1 \
    --object_background_processor random \
    --disable_flashattention \
    --train_image_encoder \
    --image_encoder_trainable_layers 2 \
    --object_types person \
    --mask_loss \
    --mask_loss_prob 0.5 \
    --object_localization \
    --object_localization_weight 1e-3 \
    --object_localization_loss balanced_l1 \
    --resume_from_checkpoint latest \
    --report_to wandb \
    # --gradient_checkpointing
