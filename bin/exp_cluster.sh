#!/bin/bash

declare -a clusters=(
    "1000 500"
    "2000 1000"
    "4000 2000"
    "6000 3000"
    "8000 4000"
    "10000 5000"
)

for cluster in "${clusters[@]}"
do
    n_query_cluster=$(echo $cluster | awk '{print $1}')
    n_doc_cluster=$(echo $cluster | awk '{print $2}')

    echo "Running experiment with n_query_cluster=$n_query_cluster, n_doc_cluster=$n_doc_cluster"    
    rye run python main.py -m \
        exp_name=exp_cluster_2 \
        generator=base_size_cluster \
        optimizer=clustered_ot_nsw \
        optimizer.params.n_doc_cluster=$n_doc_cluster \
        optimizer.params.n_query_cluster=$n_query_cluster \
        seed=0,1,2,3,4
done
