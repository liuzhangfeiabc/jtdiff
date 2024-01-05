export JT_SYNC=1
export trace_py_var=3
SAMPLE_FLAGS="--batch_size 1 --num_samples 1 --timestep_respacing ddim40 --use_ddim True"
MODEL_FLAGS="--use_ddim True --attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
model_num=`cat last_model_num.txt`
if [ "$1" != "" ]; then
    model_num="$1"
fi
echo "$SAMPLE_FLAGS $MODEL_FLAGS"
# python classifier_sample.py $MODEL_FLAGS --classifier_scale 1.0 --classifier_path trained/256x256_classifier.pt --model_path trained/256x256_diffusion.pt $SAMPLE_FLAGS
python classifier_sample.py $MODEL_FLAGS --classifier_scale 1.0 --classifier_path trained/model$model_num.pkl --model_path trained/256x256_diffusion.pt $SAMPLE_FLAGS