ROOT_DIR="/work/gg45/g45004/tf-backtrack/data" 

TASK="maze"
COMPLEXITY=9
WALL_DENSITY_LOW=0.2
WALL_DENSITY_HIGH=0.6
# /work/gg45/g45004/tf-backtrack/data/maze/n_9_wall_0.2_0.5/test.json
DATA_DIR=${ROOT_DIR}"/"${TASK}"/n_"${COMPLEXITY}"_wall_"${WALL_DENSITY_LOW}"_"${WALL_DENSITY_HIGH}
FILE_NAME="train.json"
#TASK=${ROOT_DIR}"/tasks/ED"
VOCAB_SIZE=10

#MODEL="LoopedTransformer"
#LAYER=1
#LOOP=20

MODEL="Transformer"
LAYER=6
LOOP=1

OUTPUT_DIR=${ROOT_DIR}"/output/$(basename "$TASK")_"${COMPLEXITY}"/"${MODEL}"_"${LOOP}
WANDB_NAME=$TASK"_"${COMPLEXITY}"_"${MODEL}"_"${LAYER}"_"${LOOP}

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
 --n_embd 256\
 --n_head 4\
 --n_layer ${LAYER}\
 --n_loop ${LOOP}\

# --folder ${TASK}\
# --model ${MODEL}\