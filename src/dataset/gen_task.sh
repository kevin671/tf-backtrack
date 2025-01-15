#!/bin/sh
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -g gk36
#PJM -j
#PJM --fs /work

source /work/gg45/g45004/.bashrc

conda activate base

python src/dataset/maze.py