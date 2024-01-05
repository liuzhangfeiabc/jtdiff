dir=`cat last_output_dir.txt`
num=`cat last_model_num.txt`
if [ "$1" != "" ]; then
    num="$1"
fi
if [ "$2" != "" ]; then
    dir="$2"
fi
echo "model file: output/$dir/model$num.pkl"
cp output/$dir/model$num.pkl trained 