#!/bin/bash

for i in {1..8}; do
    for j in {1..8}; do
        echo "i: $i, j: $j"
        julia --project scripts/intra-inter-cancellation/autoconverge_m_single.jl -m $i -z $j &
    done
done

wait
echo "Finished all calculations!"
