ROOT_DIR="/work/gg45/g45004/tf-backtrack/data" 

COMPLEXITY=40 # 60 # 100 
LEN_OF_FIRST_STRING=${COMPLEXITY}
DATA_DIR=${ROOT_DIR}"/ED/"${LEN_OF_FIRST_STRING}
#TASK=${ROOT_DIR}"/tasks/ED"
MAXLEN=$((2 * COMPLEXITY + 7)) # 207 # 127
MAXDATA=${MAXLEN}
NUM_RANGE=$((COMPLEXITY * 3))
VOCAB_SIZE=$((NUM_RANGE + 31))

MODEL="LoopedTransformer"
LAYER=1
LOOP=10

OUTPUT_DIR=${ROOT_DIR}"/output/$(basename "$TASK")_"${COMPLEXITY}"/"${MODEL}"_"${LOOP}
WANDB_NAME="$(basename "$TASK")_"${COMPLEXITY}"_"${MODEL}"_"${LOOP}

cd src
torchrun --standalone --nproc_per_node=1 run_exp.py\
 --data_dir ${DATA_DIR}\
 --output_dir ${OUTPUT_DIR}\
 --wandb_name ${WANDB_NAME}\
 --maxlen ${MAXLEN}\
 --maxdata ${MAXDATA}\
 --vocab ${VOCAB_SIZE}\
 --num_range ${NUM_RANGE}\
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