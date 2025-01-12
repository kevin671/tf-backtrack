ROOT_DIR="/work/gg45/g45004/tf-backtrack/data" 

TASK="countdown"
COMPLEXITY=3
DATA_DIR=${ROOT_DIR}"/b3_3_random"
FILE_NAME="train1_b3_t100_n500000_random.json"
#TASK=${ROOT_DIR}"/tasks/ED"
VOCAB_SIZE=10005

MODEL="LoopedTransformer"
LAYER=5
LOOP=20

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
 --batch_size 64\
 --epoch 100\
 --warmup 5\
 --n_embd 256\
 --n_head 4\
 --n_layer ${LAYER}\
 --n_loop ${LOOP}\

# --folder ${TASK}\
# --model ${MODEL}\