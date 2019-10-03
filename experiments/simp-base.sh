#!/bin/bash

# PATH
ROOT_DIR="/lab/maruyama/code/data-augmentation"
DATA_DIR="${ROOT_DIR}/data/simp/origin-literal/filtered/"
SAVE_DIR="${ROOT_DIR}/checkpoints/simp/origin-literal/base"
VOCAB_PATH="${ROOT_DIR}/data/vocab/bccwj-bpe.vocab"
TEST_PATH="${ROOT_DIR}/data/simp/origin-literal/filtered_test.origin"
OUTPUT_PATH="${ROOT_DIR}/outputs/origin-literal/base-transformer.literal"


# TRAINING
BATCH_SIZE=32
MAX_EPOCH=50


# OPTIMIZER, LR SCHEDULER
OPTIMIZER="adam"
LR=1e-4
LR_SCHEDULER="constant"


# MODEL
ARCH="transformer"
ENCODER_EMBED_DIM=512
DECODER_EMBED_DIM=512
ENCODER_HIDDEN_DIM=2048
DECODER_HIDDEN_DIM=2048
ENCODER_LAYERS=6
DECODER_LAYERS=6
ENCODER_HEADS=16
DECODER_HEADS=16


# DATA AUGMENTATION
AUGMENTATION="base"
SAMPLING="random"
AR_SCHEDULER="constant"
SIDE="src"
RATE=0.0


python ${ROOT_DIR}/train.py \
    --gpu \
    --train ${DATA_DIR}/filtered_train.tsv \
    --valid ${DATA_DIR}/filtered_valid.tsv \
    --savedir ${SAVE_DIR} \
    --vocab-file ${VOCAB_PATH} \
    --arch ${ARCH} \
    --batchsize ${BATCH_SIZE} \
    --max-epoch ${MAX_EPOCH} \
    --optimizer ${OPTIMIZER} \
    --lr ${LR} \
    --lr-scheduler ${LR_SCHEDULER} \
    --encoder-embed-dim ${ENCODER_EMBED_DIM} \
    --decoder-embed-dim ${DECODER_EMBED_DIM} \
    --encoder-hidden-dim ${ENCODER_HIDDEN_DIM} \
    --decoder-hidden-dim ${DECODER_HIDDEN_DIM} \
    --encoder-layers ${ENCODER_LAYERS} \
    --decoder-layers ${DECODER_LAYERS} \
    --encoder-heads ${ENCODER_HEADS} \
    --decoder-heads ${DECODER_HEADS} \
    --augmentation-strategy ${AUGMENTATION} \
    --sampling-strategy ${SAMPLING} \
    --ar-scheduler ${AR_SCHEDULER} \
    --side ${SIDE} \
    --augmentation-rate ${RATE}

python ${ROOT_DIR}/translate.py \
    --gpu \
    --model ${SAVE_DIR}/checkpoint_best.pt \
    --input ${TEST_PATH} \
    --batchsize 32 \
> ${OUTPUT_FILE}
