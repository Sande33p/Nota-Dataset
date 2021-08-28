#!/bin/bash
OUTDIR=outputs/split_CIFAR10_ResNet18/nota
mkdir -p $OUTDIR

for prob in 0.010 0.10 0.25 0.50 0.75 1.00; #
    do
    echo "Looping ... prob $prob";
        for run_num in 1 2 3 4 5;
            do
            python main.py --model ewc_on --dataset seq-cifar10-nota --load_best_args --csv_log --nota_prob ${prob} --run_num ${run_num} | tee ${OUTDIR}/ewc_on.log
            python main.py --model si --dataset seq-cifar10-nota --load_best_args --csv_log --nota_prob ${prob} --run_num ${run_num} | tee ${OUTDIR}/si.log
            python main.py --model gss --buffer_size 500 --dataset seq-cifar10-nota --load_best_args --csv_log --nota_prob ${prob} --run_num ${run_num} 2>&1 | tee ${OUTDIR}/gss.log
            python main.py --model derpp --buffer_size 500 --dataset seq-cifar10-nota --load_best_args --csv_log --nota_prob ${prob} --run_num ${run_num} 2>&1 | tee ${OUTDIR}/derpp.log       
            done
    done

