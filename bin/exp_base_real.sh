PREV_OPTIMIZERS="optimizer=greedy,greedy_nsw,expo_fair,nsw"
PROPOSED_OPTIMIZERS="optimizer=ot_nsw,pgd_nsw"
SEEDS="seed=0,1,2,3,4"

for optimizer in $PREV_OPTIMIZERS $PROPOSED_OPTIMIZERS
do
rye run python main.py -m exp_name=exp_base_real $optimizer $SEEDS generator=base_size_real
done