cat ../data/train.v1.txt ../data/train.v2.txt > ../data/train.alt.txt
cat ../data/test.v1.txt ../data/test.v2.txt > ../data/test.alt.txt

python multitask_alternate.py \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --do_eval \
  --data_dir ../data/ \
  --train_file train.alt.txt \
  --eval_file test.alt.txt \
  --num_train_epochs 5 \
  --output_dir ./alt_multi_out \
  --overwrite_output_dir
