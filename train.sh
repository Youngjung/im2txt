# Inception v3 checkpoint file.
IM2TXT="${HOME}/youngjung/im2txt"
INCEPTION_CHECKPOINT="${IM2TXT}/model/inception_v3.ckpt"

# Directory to save the model.
MODEL_DIR="${IM2TXT}/model/train"

export CUDA_VISIBLE_DEVICES=0
python im2txt/train.py \
  --input_file_pattern="${MSCOCO_DIR}/forShowAndTell/train-?????-of-00256" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}" \
  --train_inception=false \
  --number_of_steps=1000000 \
  --log_every_n_steps=10 \
  --vocab_file="${MSCOCO_DIR}/forShowAndTell/word_counts.txt"
