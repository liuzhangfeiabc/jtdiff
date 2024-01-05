source /root/anaconda3/etc/profile.d/conda.sh
conda activate jittor
run_iterations=20000
if [ "$1" != "" ]; then
    run_iterations="$1"
fi
save_interval=1000
if [ "$2" != "" ]; then
    save_interval="$2"
fi
rep=1
if [ "$REP" != "" ]; then
    rep="$REP"
fi
lr="3e-4"
if [ "$LR" != "" ]; then
    lr="$LR"
fi
for ((i=1; i<=$rep; i++))
do
RUN_ITERATIONS=$run_iterations SAVE_ITERATIONS=$save_interval LR=$lr sh ./train.sh
sh ./copy_model.sh
sh ./test.sh
sh ./img_gen.sh
done