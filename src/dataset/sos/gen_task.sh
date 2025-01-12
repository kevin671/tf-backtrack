#!/bin/bash

python countdown_generate.py --seed 4 --data_dir /work/gg45/g45004/tf-backtrack/data/b3_3_random/ --min_range 3 --start_range 3 --num_samples 500000
python countdown_generate.py --seed 4 --data_dir /work/gg45/g45004/tf-backtrack/data/b4_3_random/ --min_range 4 --start_range 4 --num_samples 500000
python countdown_generate.py --seed 4 --data_dir /work/gg45/g45004/tf-backtrack/data/b5_3_random/ --min_range 5 --start_range 5 --num_samples 500000

# python countdown_generate.py --seed 4 --data_dir /work/gg45/g45004/tf-backtrack/data/game24/ --min_range 4 --start_range 4 --num_samples 500000