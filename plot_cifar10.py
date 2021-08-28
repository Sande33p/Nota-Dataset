import numpy as np
import matplotlib
import matplotlib.pyplot as pt

algo_list = ['gss', 'derpp', 'ewc_on', 'si']
task5 = '5_tasks'
task10 = '10_tasks'

task_names = ['1.0prob', '0.75prob', '0.5prob', '0.25prob', '0.1prob', '0.01prob' ]
probs =  [1,0.75,0.5, 0.25, 0.1, 0.01]

CIL_path_to_results = 'CL_Expt/data/results/class-il/seq-cifar10-nota'
TIL_path_to_results = 'CL_Expt/data/results/task-il/seq-cifar10-nota'


def get_mean_performance(filename):
    data = np.loadtxt(filename, delimiter=",")
    return np.mean(data[-1,:])

subplots = [221, 222, 223, 224]
assay = [TIL_path_to_results, CIL_path_to_results,
    TIL_path_to_results, CIL_path_to_results]
task = [task5, task5, task10, task10]
title = ["Task incremental, 5 tasks",
         "Class incremental, 5 tasks",
         "Task incremental, 10 tasks",
         "Class incremental, 10 tasks"]

pt.figure(figsize=(8,5))

for n in range(4):
    ax = pt.subplot(subplots[n])

    y = np.zeros((4,6))

    for i, algo in enumerate(algo_list):
        for t in [task[n]]:
            for j,name in enumerate(task_names):
                for r in range(5):
                    file_i = "Acc_matrix_%d.csv" % (r+1)
                    filepath = "/".join([assay[n], algo, t, name, file_i])
                    y[i,j] += get_mean_performance(filepath)
                y[i,j] /= 5

        x = np.arange(y.shape[0])
    for j, p in enumerate(probs):
        ax.plot(x,y[:,j],"o",label=("p=" + str(p)))

    ax.set_title(title[n])
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(algo_list)
    ax.set_xlim(-0.5,3.5)
    ax.set_ylim(5,105)
    if n == 1:
        ax.legend()
    if n == 0 or n == 2:
        ax.set_ylabel("Classification accuracy")

    if n== 2 or n == 3:
        ax.set_xlabel("Algorithms")

pt.tight_layout()
pt.savefig("accuracy_cifar.pdf", dpi=300)
pt.show()



task_names = ['0.75prob', '0.5prob', '0.25prob', '0.1prob', '0.01prob' ]
probs =  [0.75,0.5, 0.25, 0.1, 0.01]

pt.figure(figsize=(8,5))

for n in range(4):
    ax = pt.subplot(subplots[n])

    y = np.zeros((4,5))

    for i, algo in enumerate(algo_list):
        for t in [task[n]]:
            for j,name in enumerate(task_names):
                for r in range(5):
                    file_i = "Acc_matrix_nota_%d.csv" % (r+1)
                    filepath = "/".join([assay[n], algo, t, name, file_i])
                    y[i,j] += get_mean_performance(filepath)
                y[i,j] /= 5

        x = np.arange(y.shape[0])
    for j, p in enumerate(probs):
        ax.plot(x,1-y[:,j]/100,"o",label=("p=" + str(p)))

    ax.set_title(title[n])
    ax.set_xticks([0,1,2,3])
    ax.set_xticklabels(algo_list)
    ax.set_xlim(-0.5,3.5)
    if n == 2:
        ax.legend()
    if n == 0 or n == 2:
        ax.set_ylabel("False positive rate")

    if n== 2 or n == 3:
        ax.set_xlabel("Algorithms")

pt.tight_layout()
pt.savefig("false_cifar.pdf", dpi=300)
pt.show()
