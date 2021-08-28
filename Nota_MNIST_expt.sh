#!/bin/bash
OUTDIR1=outputs/split_MNIST_MLP400_iter1/nota
mkdir -p $OUTDIR1

for prob in 0.010 0.10 0.25 0.50 0.75 1.0; #   
    do
    echo "Looping ... prob $prob";
        for run_num in 1 2 3 4 5;
            do
            echo "Run_num ... $run_num";
            python main.py --model si --dataset seq-mnist-nota --load_best_args --csv_log --nota_prob ${prob} --run_num ${run_num} 2>&1 | tee ${OUTDIR1}/si.log 
            python main.py --model ewc_on --dataset seq-mnist-nota --load_best_args --csv_log --nota_prob ${prob} --run_num ${run_num} 2>&1 | tee ${OUTDIR1}/ewc_on.log
            python main.py --model gss --buffer_size 500 --dataset seq-mnist-nota --load_best_args --csv_log --nota_prob ${prob} --run_num ${run_num} 2>&1 | tee ${OUTDIR1}/gss.log
            python main.py --model derpp --buffer_size 500 --dataset seq-mnist-nota --load_best_args --csv_log --nota_prob ${prob} --run_num ${run_num} 2>&1 | tee ${OUTDIR1}/derpp.log
            done
    done
