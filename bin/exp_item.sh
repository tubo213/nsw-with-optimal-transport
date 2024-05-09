PREV_OPTIMIZERS="optimizer=greedy,greedy_nsw,expo_fair,nsw"
PROPOSED_OPTIMIZERS="optimizer=ot_nsw,pg_ot_nsw"
N_DOC="generator.n_doc=50,100,200,400,800,1600"
SEEDS="seed=0,1,2,3,4"

for optimizer in $PREV_OPTIMIZERS $PROPOSED_OPTIMIZERS
do
rye run python main.py -m exp_name=exp_item $optimizer $SEEDS generator=exp_item $N_DOC
done