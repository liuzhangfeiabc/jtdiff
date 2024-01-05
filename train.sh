model_num=`cat last_model_num.txt`
if [ "$1" != "" ]; then
    model_num="$1"
fi
run_iterations=20000
if [ "$RUN_ITERATIONS" != "" ]; then
    run_iterations="$RUN_ITERATIONS"
fi
save_interval=1000
if [ "$SAVE_ITERATIONS" != "" ]; then
    save_interval="$SAVE_ITERATIONS"
fi
lr="3e-4"
if [ "$LR" != "" ]; then
    lr="$LR"
fi
TRAIN_FLAGS="--run_iterations $run_iterations --tot_iterations 500000 --save_interval $save_interval --anneal_lr True --batch_size 14 --lr $lr --weight_decay 0.05 --resume_checkpoint trained/model$model_num.pkl"
CLASSIFIER_FLAGS="--image_size 256 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
export JT_SYNC=1
export trace_py_var=3
echo "$TRAIN_FLAGS"
python classifier_train.py --data_dir ../data $TRAIN_FLAGS $CLASSIFIER_FLAGS