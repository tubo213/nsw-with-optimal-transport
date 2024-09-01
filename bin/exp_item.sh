PREV_OPTIMIZERS="optimizer=greedy_nsw"
N_DOC="generator.params.n_doc=50,100,200,400,800,1600"
SEEDS="seed=0,1,2,3,4"

for optimizer in $PREV_OPTIMIZERS
do
rye run python main.py -m exp_name=exp_item $N_DOC $optimizer generator=base_size_item $SEEDS optimizer.params.device=cpu,cuda
done