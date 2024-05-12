rye run python main.py -m \
    exp_name=exp_item \
    generator=base_size_item \
    optimizer=nsw,expo_fair \
    optimizer.params.solver=SCS \
    generator.params.n_doc=50,100,200,400 \
    seed=0,1,2,3,4