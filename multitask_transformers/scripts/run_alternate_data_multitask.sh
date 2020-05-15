cat ../data/train.v1.txt ../data/train.v2.txt > ../data/train_alt.txt
cat ../data/dev.v1.txt ../data/dev.v2.txt > ../data/dev_alt.txt

python multitask_alternate.py \
  --model_name_or_path roberta-base \
  --do_train \
  --do_eval \
  --train_batch_size 1 \
  --per_gpu_train_batch_size 1 \
  --num_train_epochs 1 \
  --output_dir .
