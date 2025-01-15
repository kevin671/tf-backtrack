#!/bin/sh
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -g gk36
#PJM -j
#PJM --fs /work

source /work/gg45/g45004/.bashrc

conda activate base

for ty in "train" "test" ; do
    rm -rf data/sat/$ty/sr5
    mkdir -p data/sat/$ty/sr5
    for i in {1..10}; do
	rm -rf data/sat/dimacs/$ty/sr5/grp$i
	mkdir -p data/sat/dimacs/$ty/sr5/grp$i
    if [ "$ty" == "train" ]; then
        n=10000  # 100k for train
    else
        n=100  # 1k for test
    fi
	python3 python src/dataset/neurosat/gen_sr_dimacs.py dimacs/$ty/sr5/grp$i $n --min_n 5 --max_n 10
    done;
done;