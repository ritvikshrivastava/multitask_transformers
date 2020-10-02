python multitask_sarc_arg.py \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --do_eval \
  --data_dir ../data/ \
  --train_file train.v1.txt \
  --eval_file test.v1.txt \
  --num_train_epochs 3 \
  --output_dir ./multi_out \
  --overwrite_output_dir
