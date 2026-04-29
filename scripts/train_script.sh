#!/bin/sh

set -e
set -x

train_file_path="$1"
config_file_path="$2"
GPU_NUM="$3"
timestamp="$4"
rest_args="${@:5}"
PORT=${PORT:-$((28500 + $RANDOM % 2000))}

if [ -z "$timestamp"  ]
then
	timestamp="`date +'%Y%m%d_%H%M%S'`"
fi

prefix="${config_file_path%%.*}"
python ${train_file_path} --dist-url="tcp://127.0.0.1:${PORT}" --num-gpus ${GPU_NUM} --resume --config-file ${config_file_path} train.output_dir=output/${prefix}_${timestamp} ${rest_args}
