ROOT_DIR="/work/gg45/g45004/tf-backtrack/data" 

TASK="sudoku"
COMPLEXITY="random"
DATA_DIR=${ROOT_DIR}"/sudoku/"${COMPLEXITY}
FILE_NAME="train.csv"
VOCAB_SIZE=10

MODEL="LoopedTransformer"
LAYER=1
LOOP=32

OUTPUT_DIR=${ROOT_DIR}"/output/$(basename "$TASK")_"${COMPLEXITY}"/"${MODEL}"_"${LOOP}
WANDB_NAME=$TASK"_"${COMPLEXITY}"_"${MODEL}"_"${LOOP}

cd src
torchrun --standalone --nproc_per_node=2 run_exp.py\
 --dataset_name ${TASK}\
 --data_dir ${DATA_DIR}\
 --file_name ${FILE_NAME}\
 --output_dir ${OUTPUT_DIR}\
 --wandb_name ${WANDB_NAME}\
 --vocab ${VOCAB_SIZE}\
 --weight_decay 0.01\
 --lr 1e-4\
 --dropout 0.0\
 --batch_size 16\
 --epoch 100\
 --warmup 5\
 --n_embd 256\
 --n_head 4\
 --n_layer ${LAYER}\
 --n_loop ${LOOP}\

# --folder ${TASK}\
# --model ${MODEL}\