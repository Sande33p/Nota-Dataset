# None-of-the-above (Nota): a continual learning benchmark

## Summary

Nota is a benchmark for continual learning algorithms. Nota is
designed to capture some of the unique challenge of online continual
learning for applications such as edge AI or real time signal processing.

The key contribution of Nota is the incorporation of **task sparsity**:
relevant samples for a specific task (such as a classification task)
are sparsely distributed in an input stream containing samples from
a *background* class that is not relevant for the task at hand. The
level of sparsity is controlled by a single parameter p, which dictates
the probability that a sample is relevant for a specific task.

A key advantage of Nota is that it can work with existing datasets,
so it is possible to design experiments such as Nota-MNIST or Nota-CIFAR100.

This repo contains the material required to run benchmarks of four
continual learning algorithms using various flavors of Nota, including
Nota-MNIST and Nota-CIFAR10.

## Authorship and Funding

Nota has been developed at Argonne National Laboratory. The authors
acknowledge support from DARPA's Lifelong Learning Machine program.
The researchers involved in the development of this benchmark are:

* Sandeep Madireddy
* Angel Yanguas-Gil

