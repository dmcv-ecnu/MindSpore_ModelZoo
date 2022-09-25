#!/usr/bin/env bash
myfile="run_eval_sysu_all_gpu.sh"

if [ ! -f "$myfile" ]; then
    echo "Please first enter DDAG_mindspore/scripts/ and run. Exit..."
    exit 0
fi

cd ..

# Note: --data-path arguments support global path or relative path(starting
#       from project root directory, i.e. /.../DDAG_mindspore/)

python eval.py \
--MSmode "GRAPH_MODE" \
--dataset SYSU \
--gpu 0 \
--device-target GPU \
--resume "XXX.ckpt" \
--tag "sysu_all_part_graph" \
--data-path "Define your own path/sysu" \
--branch main \
--sysu-mode "all" \
--part 3