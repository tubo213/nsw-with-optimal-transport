rye run python main.py -m \
    exp_name=exp_user \
    generator=base_size_user \
    optimizer=expo_fair,nsw \
    optimizer.params.solver=SCS \
    generator.n_query=50,100,200,400,800,1600 \
    seed=0,1,2,3,4