for ty in "train" "test" ; do
    rm -rf data/$ty/sr5
    mkdir -p data/$ty/sr5
    for i in {1..2}; do
	rm -rf dimacs/$ty/sr5/grp$i
	mkdir -p dimacs/$ty/sr5/grp$i
	python3 gen_sr_dimacs.py dimacs/$ty/sr5/grp$i 10 --min_n 5 --max_n 10
    done;
done;