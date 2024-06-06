OPTIMIZERS="optimizer=ot_nsw,pgd_nsw, optimizer.params.device=cpu"
N_DOC="generator.params.n_doc=50,100,200,400,800,1600"
N_QUERY="generator.params.n_query=50,100,200,400,800,1600"
SEEDS="seed=0,1,2,3,4"


rye run python main.py -m exp_name=exp_item $N_DOC $optimizer generator=base_size_item $SEEDS

rye run python main.py -m \
    exp_name=exp_item \
    generator=base_size_item \
    optimizer=ot_nsw \
    optimizer.params.device=cpu \
    generator.params.n_doc=50,100,200,400,800,1600 \
    seed=0,1,2,3,4

rye run python main.py -m \
    exp_name=exp_user \
    generator=base_size_user \
    optimizer=ot_nsw \
    optimizer.params.device=cpu \
    generator.params.n_query=50,100,200,400,800,1600 \
    seed=0,1,2,3,4