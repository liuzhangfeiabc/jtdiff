dir=`cat last_output_dir.txt`
if [ "$1" != "" ]; then
    dir="$1"
fi
echo "npz file: output/$dir/samples_1x256x256x3.npz"
python scripts/sample_process.py --image_file output/$dir/samples_1x256x256x3.npz --image_dir images