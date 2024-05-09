PREV_OPTIMIZERS="optimizer=greedy,greedy_nsw,expo_fair,nsw"
PROPOSED_OPTIMIZERS="optimizer=ot_nsw,pg_ot_nsw"
SEEDS="seed=0,1,2,3,4,5,6,7,8,9"

# run
for generater in "size1" "size2" "size3" "size4" "size5" "size6"
do
    for optimizer in $PREV_OPTIMIZERS $PROPOSED_OPTIMIZERS
    do
    rye run python main.py -m exp_name=Varying_data_size $optimizer $SEEDS generator=$generater
    done
done