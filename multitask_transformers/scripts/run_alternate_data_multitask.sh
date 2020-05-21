cat ../data/train.v1.txt ../data/train.v2.txt > ../data/train.alt.txt
cat ../data/dev.v1.txt ../data/dev.v2.txt > ../data/dev.alt.txt

python multitask_alternate.py \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --do_eval \
  --data_dir ../data/ \
  --train_file train.alt.txt \
  --eval_file dev.alt.txt \
  --num_train_epochs 1 \
  --output_dir .
