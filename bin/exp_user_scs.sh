rye run python main.py -m \
    exp_name=exp_user \
    generator=base_size_user \
    optimizer=nsw,expo_fair \
    optimizer.params.solver=SCS \
    generator.params.n_query=50,100,200,400 \
    seed=0,1,2,3,4