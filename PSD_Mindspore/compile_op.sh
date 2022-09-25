cd ext/nearest_neighbors
python setup.py install --home="."
cd ../../

cd ext/cpp_wrappers
sh compile_wrappers.sh
cd ../../../