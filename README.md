# ClusterNet

This code implements and evaluates the ClusterNet method described in the NeurIPS 2019 [paper](https://arxiv.org/abs/1905.13732) "End to End Learning and Optimization on Graphs". ClusterNet provides a differentiable k-means clustering layer which is used as a building block for solving graph optimization problems. 

```
@inproceedings{wilder2019end,
  title={End to End Learning and Optimization on Graphs},
  author={Wilder, Bryan and Ewing, Eric and Dilkina, Bistra and Tambe, Milind},
  booktitle={Advances in Neural and Information Processing Systems},
  year={2019}
}
```

# Files

* experiments_singlegraph.py runs the experiments which do link prediction on a given graph. See below for parameters.
* experiments_inductive.py runs the experiments which evaluate training and generalization on a distribution of graphs.
* models.py contains the definitions for both the ClusterNet model (GCNClusterNet) as well as various models used for the baselines.
* modularity.py contains helper functions and baseline optimization algorithms for the community detection task.
* kcenter.py contains helper functions and baseline optimization algorithms for the facility location task.
* loss_functions.py contains definitions of the loss function used to train ClusterNet and GCN-e2e for both tasks.
* utils.py contains helper functions to load/manipulate the datasets.

# Datasets

Included are the datasets used to run the experiments in the paper. Here are the mappings between the filenames (which can be used as the --dataset argument to the code) and the names of the datasets in the paper:

* cora: the [cora](https://relational.fit.cvut.cz/dataset/CORA) dataset (cora)
* citeseer: the [citeseer](https://linqs.soe.ucsc.edu/data) dataset (citeseer)
* moreno: the [adolescent social network](http://konect.uni-koblenz.de/networks/moreno_health) (adol)
* protein_vidal: the [protein interaction network](http://konect.uni-koblenz.de/networks/maayan-vidal) (protein)
* fb_small: the [facebook network](http://konect.uni-koblenz.de/networks/ego-facebook) (fb)
* pubmed: the [pubmed](https://linqs.soe.ucsc.edu/data) citation dataset (pubmed)
* synthetic_spa: a [synthetic distribution](https://dl.acm.org/citation.cfm?id=3237383.3237507) based on spacial preferential attachment model (synthetic)

# Arguments to experiments scripts

The experiments_singlegraph.py and experiments_inductive.py scripts take a number of arguments to specify the experiment to run as well as various hyperparameters. The following arguments determine which experiment is run. The remainder specify standard hyperparameters for the GCNs/training process, with default values from the paper provided in the examples below.

* objective: this determines which optimization problem is used for the experiment. Use "--objective modularity" for the community detection task and "--objective kcenter" for the facility location task. 
* dataset: which dataset to run on. For the datasets in the paper, which are included in this release, see the section below. Note that for experiments on the facility location problem, you should use the "datasetname_connected", which retains only the largest connected component to ensure that distances are well-defined. The only exception is the synthetic graphs, which are always connected. 
* pure_opt: Add the flag "--pure_opt" to run experiments with no link prediction step, i.e., all methods observe the full graph. This is disabled by default.

# Examples of running the experiments

Example running the single-graph experiment for the community detection problem on the cora dataset:

```
python experiments_singlegraph.py --objective modularity --hidden 50 --embed_dim 50 --weight_decay 5e-4 --dropout 0.2 --train_iters 1001 --clustertemp 50 --num_cluster_iter 1 --lr 0.01 --dataset cora
```

For the kcenter problem, make sure to use that corresponding "_connected" version of the graph, which keeps only the largest connected component so that distances are well-defined:

~~~
python experiments_singlegraph.py --objective kcenter --hidden 50 --embed_dim 50 --weight_decay 5e-4 --dropout 0.2 --train_iters 1001 --clustertemp 30 --num_cluster_iter 1 --lr 0.01 --dataset cora_connected
~~~

The process is the same for the inductive experiments, e.g.,:

~~~
python experiments_inductive.py --objective modularity --hidden 50 --embed_dim 50 --weight_decay 5e-4 --dropout 0.2 --train_iters 220 --clustertemp 70 --num_cluster_iter 1 --lr 0.001 --dataset pubmed
~~~

# Dependencies

The Dockerfile in the main directory builds the environment that was used to run the original experiments, with the exception of [pygcn](https://github.com/tkipf/pygcn/tree/master/pygcn), which needs to be downloaded and installed separately. For reference, the individual dependencies are:

* PyTorch (tested on version 1.2)
* networkx (tested on version 2.3)
* igraph (tested on version 0.7.1). This is optional; only used to accelerate pre-processing operations.
* [pygcn](https://github.com/tkipf/pygcn/tree/master/pygcn)
* sklearn (tested on version 0.21.3)
* numpy (tested on version 1.16.5)
