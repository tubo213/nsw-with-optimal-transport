#!/bin/bash

declare -a datasizes=(
    "3125 625"
    "6250 1250"
    "12500 2500"
    "25000 5000"
)

declare -a rates=(0.1 0.4 0.6)

for datasize in "${datasizes[@]}"
do
    n_query=$(echo $datasize | awk '{print $1}')
    n_doc=$(echo $datasize | awk '{print $2}')
    
    for r in "${rates[@]}"
    do
        n_query_cluster=$(echo "$n_query * $r" | bc)
        n_doc_cluster=$(echo "$n_doc * $r" | bc)
        
        # intに変換
        n_query_cluster=${n_query_cluster%.*}
        n_doc_cluster=${n_doc_cluster%.*}

        echo "Running experiment with n_query=$n_query, n_doc=$n_doc, n_query_cluster=$n_query_cluster, n_doc_cluster=$n_doc_cluster, rate=$r" 
        rye run python main.py -m \
            exp_name=exp_cluster_datasize_2 \
            generator=base_size_cluster \
            generator.params.n_query=$n_query \
            generator.params.n_doc=$n_doc \
            optimizer=clustered_ot_nsw \
            optimizer.params.n_doc_cluster=$n_doc_cluster \
            optimizer.params.n_query_cluster=$n_query_cluster \
            seed=0,1,2,3,4
        done
done
