CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/pretrained_model/chinese_roberta_wwm_ext_pytorch
export DATA_DIR=$CURRENT_DIR/BERT-NER-Pytorch/datasets
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="aigo"
mkdir $OUTPUR_DIR
#
python $CURRENT_DIR/BERT-NER-Pytorch/run_ner_crf.py \
  --model_type=bert \
  --model_name_or_path=$BERT_BASE_DIR \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=16 \
  --learning_rate=3e-5 \
  --crf_learning_rate=5e-3 \
  --num_train_epochs=15.0 \
  --logging_steps=-1 \
  --save_steps=-1 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --markup=bio \
  --seed=42


