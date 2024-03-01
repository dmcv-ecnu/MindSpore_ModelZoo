#!/bin/sh
# shellcheck disable=SC2164

cd ./nearest_neighbors
python setup.py install --home="."
cd ../

cd ./cpp_wrappers
sh compile_wrappers.sh