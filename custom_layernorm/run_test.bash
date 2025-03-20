#!/bin/bash

batch_sizes=(1 2 4 8 16 32 64)
seq_lens=(128 256 512 1024 2048 4096)

for batch_size in "${batch_sizes[@]}"; do
    for seq_len in "${seq_lens[@]}"; do
        echo "Testing batch_size=${batch_size}, seq_len=${seq_len}"
        ncu --metrics sm__throughput.avg,sm__sass_thread_inst_executed_op_shared_ld.sum,sm__sass_thread_inst_executed_op_shared_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum -- python test.py --batch_size ${batch_size} --seq_len ${seq_len}
    done
done