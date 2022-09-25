#!/usr/bin/env bash
myfile="run_eval_sysu_indoor_ascend.sh"

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
--device-id 0 \
--device-target Ascend \
--resume "XXX.ckpt" \
--tag "sysu_indoor_part_graph" \
--data-path "Define your own path/sysu" \
--branch main \
--sysu-mode "indoor" \
--part 3