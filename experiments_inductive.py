from utils import load_data
import torch
import argparse
import numpy as np
import torch.optim as optim
import random
import pickle
import torch.nn as nn
import sklearn
from modularity import greedy_modularity_communities, partition, baseline_spectral, make_modularity_matrix
from utils import make_normalized_adj, edge_dropout, negative_sample
from models import GCNLink, GCNClusterNet, GCNDeep, GCNDeepSigmoid, GCN
from loss_functions import loss_kcenter, loss_modularity
import copy
from kcenter import CenterObjective, make_all_dists, gonzalez_kcenter, greedy_kcenter, make_dists_igraph, rounding

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=24, help='Random seed.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=50,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--embed_dim', type=int, default=50,
                    help='Dimensionality of node embeddings')
parser.add_argument('--K', type=int, default=5,
                    help='How many partitions')
parser.add_argument('--negsamplerate', type=int, default=1,
                    help='How many negative examples to include per positive in link prediction training')
parser.add_argument('--edge_dropout', type=float, default=0.2,
                    help='Rate at which to remove edges in link prediction training')
parser.add_argument('--objective', type=str, default='kcenter',
                    help='What objective to optimize (currently partitioning or modularity')
parser.add_argument('--dataset', type=str, default='synthetic_spa',
                    help='which network to load')
parser.add_argument('--clustertemp', type=float, default=20,
                    help='how hard to make the softmax for the cluster assignments')
parser.add_argument('--kcentertemp', type=float, default=100,
                    help='how hard to make seed selection softmax assignment')
parser.add_argument('--kcentermintemp', type=float, default=0,
                    help='how hard to make the min over nodes in kcenter training objective')
parser.add_argument('--use_igraph', action='store_true', default=True, help='use igraph to compute shortest paths in twostage kcenter')
parser.add_argument('--train_iters', type=int, default=1000,
                    help='number of training iterations')
parser.add_argument('--num_cluster_iter', type=int, default=1,
                    help='number of iterations for clustering')
parser.add_argument('--singletrain', action='store_true', default=False, help='only train on a single instance')
parser.add_argument('--pure_opt', action='store_true', default=False, help='do only optimization, no link prediction needed')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    

if 'synthetic_spa' not in args.dataset:
    directory = args.dataset
else:
    directory = 'synthetic_spa'
# Load data
reload_data = True
pure_optimization = args.pure_opt
train_pct = 0.4

if 'synthetic' in args.dataset:
    num_graphs = 60
    numtest = 30
else: #pubmed dataset
    num_graphs = 20
    numtest = 8
    
if reload_data:
    bin_adj_all = []
    adj_all = []
    #features = []
    adj_train = []
    bin_adj_train = []
    features_train = []
    features_all = []
    dist_all = []
    dist_train = []
    for i in range(num_graphs):
        adj_i, features_i, labels_i, idx_train, idx_val, idx_test = load_data('data/{}/'.format(directory), '{}_{}'.format(args.dataset, i))
        bin_adj_i = (adj_i.to_dense() > 0).float()
        bin_adj_all.append(bin_adj_i)
        adj_all.append(adj_i.coalesce())
        features_all.append(features_i)
        adj_train_i, features_train_i, labels_train_i, idx_train, idx_val, idx_test = load_data('data/{}/'.format(directory), '{0}_{1}_train_{2:.2f}'.format(args.dataset, i, train_pct))
        bin_adj_train_i = (adj_train_i.to_dense() > 0).float()
        bin_adj_train.append(bin_adj_train_i)
        adj_train.append(adj_train_i.coalesce())
        features_train.append(features_train_i)


vals = {}
algs = ['ClusterNet', 'ClusterNet-ft', 'ClusterNet-ft-only', 'GCN-e2e', 'GCN-e2e-ft', 'GCN-e2e-ft-only']
if args.objective == 'modularity':
    ts_algos = ['agglomerative', 'recursive', 'spectral']
elif args.objective == 'kcenter':
    ts_algos = ['gonzalez', 'greedy']
for algo in ts_algos:
    algs.append('train-' + algo)
    algs.append('ts-' + algo)
    algs.append('ts-ft-' + algo)
    algs.append('ts-ft-only-' + algo)
for algo in algs:
    vals[algo] = np.zeros(numtest)

aucs_algs = ['ts', 'ts-ft', 'ts-ft-only']
aucs = {}
for algo in aucs_algs:
    aucs[algo] = np.zeros(numtest)

if args.objective == 'modularity':
    mods_test = [make_modularity_matrix(A) for A in bin_adj_all]
    mods_train = [make_modularity_matrix(A) for A in bin_adj_train]
    test_object = mods_test
    train_object = mods_train
    loss_fn = loss_modularity
elif args.objective == 'kcenter':
    for i in range(num_graphs):
        try:
            dist_all.append(torch.load('{}_{}_test_dist.pt'.format(args.dataset, i)))
            dist_train.append(torch.load('{}_{}_{:.2f}_train_dist.pt'.format(args.dataset, i, train_pct)))
            diameter = dist_all[-1].max()
        except:
            dist_all_i = make_all_dists(bin_adj_all[i], 100)
            diameter = dist_all_i[dist_all_i < 100].max()
            dist_all_i[dist_all_i == 100] = diameter
            torch.save(dist_all_i, '{}_{}_test_dist.pt'.format(args.dataset, i))
            dist_all.append(dist_all_i)
            dist_train_i = make_all_dists(bin_adj_train[i], 100)
            dist_train_i[dist_train_i == 100] = diameter
            torch.save(dist_train_i, '{}_{}_{:.2f}_train_dist.pt'.format(args.dataset, i, train_pct))
            dist_train.append(dist_train_i)
    
    obj_train = [CenterObjective(d, diameter, args.kcentermintemp) for d in dist_train]
    obj_train_hardmax = [CenterObjective(d, diameter, args.kcentermintemp, hardmax=True) for d in dist_train]
    obj_test = [CenterObjective(d, diameter, args.kcentertemp, hardmax=True) for d in dist_all]
    obj_test_softmax = [CenterObjective(d, diameter, args.kcentermintemp) for d in dist_all]
    
    test_object = obj_test
    train_object = obj_train
    loss_fn = loss_kcenter


if pure_optimization:
    train_object = test_object
    adj_train = adj_all
    bin_adj_train = bin_adj_all
    dist_train = dist_all

for test_idx in range(1):
    if 'pubmed' in args.dataset:
        valid_instances = list(range(10, 12))
        test_instances= list(range(12, 20))
    if 'synthetic' in args.dataset:
        test_instances = list(range(20, 50))
        valid_instances = list(range(50, 60))
    if not args.singletrain:
        train_instances = [x for x in range(num_graphs) if x not in test_instances and x not in valid_instances]
    else:
        train_instances = [0]
        
    nfeat = features_all[0].shape[1]
    
    K = args.K
    # Model and optimizer
    model_ts = GCNLink(nfeat=nfeat,
                nhid=args.hidden,
                nout=args.embed_dim,
                dropout=args.dropout)
    
    model_cluster = GCNClusterNet(nfeat=nfeat,
                nhid=args.hidden,
                nout=args.embed_dim,
                dropout=args.dropout,
                K = args.K, 
                cluster_temp = args.clustertemp)
    
    #keep a couple of initializations here so that the random seeding lines up
    #with results reported in the paper -- removing these is essentially equivalent to 
    #changing the seed
    _ = GCN(nfeat, args.hidden, args.embed_dim, args.dropout)
    _ = nn.Parameter(torch.rand(K, args.embed_dim))

#    
    if args.objective == 'modularity':
        model_gcn = GCNDeep(nfeat=nfeat,
                    nhid=args.hidden,
                    nout=args.K,
                    dropout=args.dropout, 
                    nlayers=2)
    elif args.objective == 'kcenter':
            model_gcn = GCNDeepSigmoid(nfeat=nfeat,
                nhid=args.hidden,
                nout=1,
                dropout=args.dropout, 
                nlayers=2)

    
    optimizer = optim.Adam(model_cluster.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    
    losses = []
    losses_test = []
    num_cluster_iter = args.num_cluster_iter
        
    def get_average_loss(model, adj, bin_adj, bin_adj_for_loss, objectives, instances, features, num_reps = 10, hardmax = False, update = False, algoname =  None):
        if hardmax:
            model.eval()
        loss = 0
        for _ in range(num_reps):
            for idx, i in enumerate(instances):
                mu, r, embeds, dist = model(features[i], adj[i], num_cluster_iter)
                if hardmax:
                    r = torch.softmax(100*r, dim=1)
                this_loss = loss_fn(mu, r, embeds, dist, bin_adj_for_loss[i], objectives[i], args)
                loss += this_loss
                if update:
                    vals[algoname][test_instances.index(i)] = this_loss.item()
        if hardmax:
            model.train()
        return loss/(len(instances)*num_reps)
    
    
    def get_kcenter_test_loss(model, adj, bin_adj, train_objectives, test_objectives, instances, features, num_reps = 10, hardmax = False, update = False, algoname = None):
        loss = 0
        for idx, i in enumerate(instances):
            best_loss = 100
            x_best = None
            for _ in range(num_reps):
                mu, r, embeds, dist = model(features[i], adj[i], num_cluster_iter)
                x = torch.softmax(dist*args.kcentertemp, 0).sum(dim=1)
                x = 2*(torch.sigmoid(4*x) - 0.5)
                if x.sum() > args.K:
                    x = args.K*x/x.sum()
                train_loss = loss_fn(mu, r, embeds, dist, bin_adj[i], train_objectives[i], args)
                if train_loss.item() < best_loss:
                    best_loss = train_loss.item()
                    x_best = x
            testvals = []; trainvals = []
            for _ in range(50):
                y = rounding(x_best)
                testvals.append(test_objectives[i](y).item())
                trainvals.append(train_objectives[i](y).item())
            loss += testvals[np.argmin(trainvals)]
            if update:
                vals[algoname][test_instances.index(i)] = testvals[np.argmin(trainvals)]
        return loss/(len(instances))


    #Decision-focused training
    if True:
        for t in range(args.train_iters):
            i = np.random.choice(train_instances)
            mu, r, embeds, dist = model_cluster(features_train[i], adj_train[i], num_cluster_iter)
            if args.objective == 'modularity':
                loss = loss_fn(mu, r, embeds, dist, bin_adj_all[i], test_object[i], args)
            else:
                loss = loss_fn(mu, r, embeds, dist, bin_adj_all[i], obj_test_softmax[i], args)
            if args.objective != 'kcenter':
                loss = -loss
            optimizer.zero_grad()
            loss.backward()
            if t % 100 == 0 and t != 0:
                num_cluster_iter = 5
            if t % 10 == 0:
                if args.objective == 'modularity':
                    r = torch.softmax(100*r, dim=1)
                loss_train = get_average_loss(model_cluster, adj_train, bin_adj_train, bin_adj_all, test_object, train_instances, features_train, hardmax=True)
                loss_test = get_average_loss(model_cluster, adj_train, bin_adj_train, bin_adj_all, test_object, test_instances, features_train, hardmax=True)
                loss_valid = get_average_loss(model_cluster, adj_train, bin_adj_train, bin_adj_all, test_object, valid_instances, features_train, hardmax=True)
                losses_test.append(loss_test.item())
                print(t, loss_train.item(), loss_test.item(), loss_valid.item())
            losses.append(loss.item())
            optimizer.step()
        if args.objective == 'kcenter':
            loss_round = get_kcenter_test_loss(model_cluster, adj_train, bin_adj_train, train_object, test_object, test_instances, features_train, update = True, algoname = 'ClusterNet')
        elif args.objective == 'modularity':
            loss_test = get_average_loss(model_cluster, adj_train, bin_adj_train, bin_adj_all, test_object, test_instances, features_train, hardmax=True, update = True, algoname = 'ClusterNet')
        print('after training', np.mean(vals['ClusterNet'][:numtest]), np.std(vals['ClusterNet']))
        if args.singletrain:
            pickle.dump((vals, aucs), open('results_distributional_singletrain_{}_{}_{}.pickle'.format(args.dataset, args.objective, args.K), 'wb'))
            break
        def fine_tune(model, features, adj, bin_adj, objective, num_training_iters = 1000):
            optimizer = optim.Adam(model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)
            for t in range(num_training_iters):
                mu, r, embeds, dist = model(features, adj, num_cluster_iter)
                loss = loss_fn(mu, r, embeds, dist, bin_adj, objective, args)
                if args.objective != 'kcenter':
                    loss = -loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        num_cluster_iter = 1
        loss_finetune = 0
        loss_round = 0
        for i in test_instances:
            model_i = copy.deepcopy(model_cluster)
            fine_tune(model_i, features_train[i], adj_train[i], bin_adj_train[i], train_object[i], num_training_iters = 50)
            loss_finetune += get_average_loss(model_i, adj_train, bin_adj_train, bin_adj_all, test_object, [i], features_train, hardmax=True, update = True, algoname = 'ClusterNet-ft').item()
            if args.objective == 'kcenter':
                loss_round += get_kcenter_test_loss(model_i, adj_train, bin_adj_train, train_object, test_object, [i], features_train, update = True, algoname = 'ClusterNet-ft')
    
        print('finetune', np.mean(vals['ClusterNet-ft']), np.std(vals['ClusterNet-ft']))
        loss_finetune = 0
        loss_round = 0
        for i in test_instances:
            model_i = GCNClusterNet(nfeat=nfeat,
                    nhid=args.hidden,
                    nout=args.embed_dim,
                    dropout=args.dropout,
                    K = args.K, 
                    cluster_temp = args.clustertemp)
            fine_tune(model_i, features_train[i], adj_train[i], bin_adj_train[i], train_object[i], num_training_iters = 500)
            loss_finetune += get_average_loss(model_i, adj_train, bin_adj_train, bin_adj_all, test_object, [i], features_train, hardmax=True, update = True, algoname = 'ClusterNet-ft-only').item()
            if args.objective == 'kcenter':
                loss_round += get_kcenter_test_loss(model_i, adj_train, bin_adj_train, train_object, test_object, [i], features_train, update = True, algoname = 'ClusterNet-ft-only')
    
        print('finetune only', np.mean(vals['ClusterNet-ft-only']), np.std(vals['ClusterNet-ft-only']))
    
    #Train a two-stage model for link prediction with cross-entropy loss and 
    #negative sampling
        
    def train_twostage(model_ts, train_instances, test_instances, features, algoname):
        optimizer_ts = optim.Adam(model_ts.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
        edges = {}
        edges_eval = {}
        labels_eval = {}
        for i in train_instances + test_instances:
            edges[i] = adj_train[i].indices().t()
            edges_eval_i, labels_eval_i = negative_sample(adj_all[i].indices().t(), 1, bin_adj_all[i])
            edges_eval[i] = edges_eval_i
            labels_eval[i] = labels_eval_i
        
        def get_evaluation(instances):
            test_ce = 0
            test_auc = 0
            for i in instances:
                preds_test_eval = model_ts(features[i], adj_train[i], edges_eval[i])
                test_ce += torch.nn.BCEWithLogitsLoss()(preds_test_eval, labels_eval[i])
                test_auc_i = sklearn.metrics.roc_auc_score(labels_eval[i].long().detach().numpy(), nn.Sigmoid()(preds_test_eval).detach().numpy())
                aucs[algoname][test_instances.index(i)] = test_auc
                test_auc += test_auc_i
            return test_ce/len(instances), test_auc/len(instances)
        
        for t in range(150):
            i = np.random.choice(train_instances)
            adj_input = make_normalized_adj(edge_dropout(edges[i], args.edge_dropout), bin_adj_train[i].shape[0])
            edges_eval_i, labels_i = negative_sample(edges[i], args.negsamplerate, bin_adj_train[i])
            preds = model_ts(features[i], adj_input, edges_eval_i)
            loss = torch.nn.BCEWithLogitsLoss()(preds, labels_i)
            optimizer_ts.zero_grad()
            loss.backward()
            if t % 10 == 0:
                test_ce, test_auc = get_evaluation(test_instances)
                print(t, loss.item(), test_ce.item(), test_auc)
            optimizer_ts.step()
    
    
    def test_twostage(model_ts, test_instances_eval, algoname):
        for test_i in test_instances_eval:
            #predict probability that all unobserved edges exist
            n = adj_train[test_i].shape[0]
            indices = torch.tensor(np.arange(n))
            to_pred = torch.zeros(n**2, 2)
            to_pred[:, 1] = indices.repeat(n)
            for i in range(n):
                to_pred[i*n:(i+1)*n, 0] = i
            to_pred = to_pred.long()
            preds = model_ts(features_train[test_i], adj_train[test_i], to_pred)
            preds = nn.Sigmoid()(preds).view(n, n)
            preds = bin_adj_train[test_i] + (1 - bin_adj_train[test_i])*preds
            
            if args.objective == 'modularity':
                r = greedy_modularity_communities(preds, K)
                loss = loss_fn(None, r, None, None, bin_adj_all[test_i], test_object[test_i], args).item()
                vals[algoname + '-agglomerative'][test_instances.index(test_i)] = loss
                r = partition(preds, K)
                loss = loss_fn(None, r, None, None, bin_adj_all[test_i], test_object[test_i], args).item()
                vals[algoname + '-recursive'][test_instances.index(test_i)] = loss
                degrees = preds.sum(dim=1)
                preds = torch.diag(1./degrees)@preds
                mod_pred = make_modularity_matrix(preds)
                r = baseline_spectral(mod_pred, K)
                loss = loss_fn(None, r, None, None, bin_adj_all[test_i], test_object[test_i], args).item()
                vals[algoname + '-spectral'][test_instances.index(test_i)] = loss
            elif args.objective == 'kcenter':
                print('making dists')
                if args.use_igraph:
                    dist_ts =  make_dists_igraph(preds)
                else:
                    dist_ts = make_all_dists(preds, 100)
                    diameter = dist_ts[dist_ts < 100].max()
                    dist_ts[dist_ts == 100] = diameter
                print(test_i)
                dist_ts = dist_ts.float()
                diameter = dist_ts.max()
                x = gonzalez_kcenter(dist_ts, K)
                loss = obj_test[test_i](x)
                vals[algoname + '-gonzalez'][test_instances.index(test_i)] = loss.item()
                x = greedy_kcenter(dist_ts, diameter, K)
                loss = obj_test[test_i](x)
                vals[algoname + '-greedy'][test_instances.index(test_i)] = loss.item()
        
    if True:
        print('two stage')
        #do pretrained two stage
        train_twostage(model_ts, train_instances, test_instances, features_train, 'ts')
        test_twostage(model_ts, test_instances, 'ts')
        for algo in algs:
            if 'ts' in algo and 'ft' not in algo:
                print(algo, np.mean(vals[algo]), np.std(vals[algo]))
        
#        do finetuning
        loss_agglom_ft = 0; loss_recursive_ft = 0; loss_spectral_ft = 0
        loss_greedy_ft = 0; loss_gonzalez_ft = 0
        for i in test_instances:
            model_i = copy.deepcopy(model_ts)
            train_twostage(model_i, [i], [i], features_train, 'ts-ft')
            test_twostage(model_ts, [i], 'ts-ft')
        for algo in algs:
            if 'ts-ft' in algo and 'only' not in algo:
                print(algo, np.mean(vals[algo]), np.std(vals[algo]))
        
        #do only finetuning
        loss_agglom_ft_only = 0; loss_recursive_ft_only = 0; loss_spectral_ft_only = 0
        loss_greedy_ft_only = 0; loss_gonzalez_ft_only = 0
        for i in test_instances:
            model_i = GCNLink(nfeat=nfeat,
                nhid=args.hidden,
                nout=args.embed_dim,
                dropout=args.dropout)
            train_twostage(model_i, [i], [i], features_train, 'ts-ft-only')
            test_twostage(model_ts, [i], 'ts-ft-only')
        for algo in algs:
            if 'ts-ft-only' in algo:
                print(algo, np.mean(vals[algo]), np.std(vals[algo]))
    
    if True:
        def get_average_loss(model, adj, bin_adj, bin_adj_for_loss, objectives, instances, features, num_reps = 1, hardmax = False, update = False, algoname = None):
            loss = 0
            for _ in range(num_reps):
                for i in instances:
                    if args.objective == 'modularity':
                        r = model(features[i], adj[i])
                        r = torch.softmax(r, dim = 1)
                        if hardmax:
                            r = torch.softmax(100*r, dim=1)
                        this_loss = -loss_fn(None, r, None, None, bin_adj_for_loss[i], objectives[i], args)
                    elif args.objective == 'kcenter':
                        x = model(features[i], adj[i])
                        if x.sum() > K:
                            x = K*x/x.sum()
                        this_loss = objectives[i](x)
                    loss += this_loss
                    if update:
                        vals[algoname][test_instances.index(i)] = this_loss.item()
            return loss/(len(instances)*num_reps)
        
        
        def get_kcenter_test_loss(model, adj, bin_adj, train_objectives, test_objectives, instances, features, num_reps = 10, hardmax = False, update = False, algoname = None):
            loss = 0
            for i in instances:
                best_loss = 100
                x_best = None
                for _ in range(num_reps):
                    x = model(features[i], adj[i])
                    if x.sum() > args.K:
                        x = args.K*x/x.sum()
                    train_loss = train_objectives[i](x)
                    if train_loss.item() < best_loss:
                        best_loss = train_loss.item()
                        x_best = x
                testvals = []; trainvals = []
                for _ in range(50):
                    y = rounding(x_best)
                    testvals.append(test_objectives[i](y).item())
                    trainvals.append(train_objectives[i](y).item())
                loss += testvals[np.argmin(trainvals)]
                if update:
                    vals[algoname][test_instances.index(i)] = testvals[np.argmin(trainvals)]
            return loss/(len(instances))

        print('just GCN')
        optimizer_gcn = optim.Adam(model_gcn.parameters(), lr = args.lr, 
                                   weight_decay = args.weight_decay)
        
        def train_gcn_model(model, train_instances, num_iters = 1000, verbose=False):
            for t in range(num_iters):
                i = random.choice(train_instances)
                loss = get_average_loss(model_gcn, adj_train, bin_adj_train, bin_adj_train, train_object, [i], features_train)
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
                if t % 100 == 0 and verbose:
                    loss_train = get_average_loss(model_gcn, adj_all, bin_adj_all, bin_adj_all, test_object, train_instances, features_all)
                    loss_test = get_average_loss(model_gcn, adj_train, bin_adj_train, bin_adj_all, test_object, test_instances, features_train, hardmax=True)
                    losses_test.append(loss_test.item())
                    print(t, loss.item(), loss_train.item(), loss_test.item())
                    
        train_gcn_model(model_gcn, train_instances, num_iters = 1000, verbose = True)
        if args.objective == 'kcenter':
            loss_round = get_kcenter_test_loss(model_gcn, adj_train, bin_adj_train, train_object, test_object, test_instances, features_train , update = True, algoname = 'GCN-e2e')
        elif args.objective == 'modularity':
            loss_gcne2e = get_average_loss(model_gcn, adj_train, bin_adj_train, bin_adj_all, test_object, test_instances, features_train, hardmax=True, update = True, algoname = 'GCN-e2e').item()
        print('GCN-e2e', np.mean(vals['GCN-e2e']), np.std(vals['GCN-e2e']))
        
        #################
        #GCN FINETUNE
        #################
        loss_finetune = 0
        loss_round = 0
        for i in test_instances:
            model_i = copy.deepcopy(model_gcn)
            train_gcn_model(model_i, [i], num_iters = 500)
            loss_finetune += get_average_loss(model_i, adj_train, bin_adj_train, bin_adj_all, test_object, [i], features_train, hardmax=True, update = True, algoname = 'GCN-e2e-ft').item()
            if args.objective == 'kcenter':
                loss_round += get_kcenter_test_loss(model_i, adj_train, bin_adj_train, train_object, test_object, [i], features_train, update = True, algoname = 'GCN-e2e-ft')
        print('GCN-e2e-ft', np.mean(vals['GCN-e2e-ft']), np.std(vals['GCN-e2e-ft']))

        ######################
        #GCN ONLY FINETUNE
        ######################
        loss_finetune = 0
        loss_round = 0
        for i in test_instances:
            if args.objective != 'kcenter':
                model_i = GCNDeep(nfeat=nfeat,
                    nhid=args.hidden,
                    nout=args.K,
                    dropout=args.dropout,
                    nlayers=2)
            else:
                model_i = GCNDeepSigmoid(nfeat=nfeat,
                    nhid=args.hidden,
                    nout=args.K,
                    dropout=args.dropout,
                    nlayers=2)
            train_gcn_model(model_i, [i], num_iters = 500)
            loss_finetune += get_average_loss(model_i, adj_train, bin_adj_train, bin_adj_all, test_object, [i], features_train, hardmax=True, update = True, algoname = 'GCN-e2e-ft-only').item()
            if args.objective == 'kcenter':
                loss_round += get_kcenter_test_loss(model_i, adj_train, bin_adj_train, train_object, test_object, [i], features_train, update = True, algoname = 'GCN-e2e-ft-only')
    
        print('GCN-e2e-ft-only', np.mean(vals['GCN-e2e-ft-only']), np.std(vals['GCN-e2e-ft-only']))

    
    if True:
        print('only training edges')
        for idx, i in enumerate(test_instances):
            if args.objective == 'modularity':
                preds = bin_adj_train[i]
                r = greedy_modularity_communities(preds, K)
                vals['train-agglomerative'][idx] = loss_fn(None, r, None, None, bin_adj_all[i], test_object[i], args).item()
                r = partition(preds, K)
                vals['train-recursive'][idx] = loss_fn(None, r, None, None, bin_adj_all[i], test_object[i], args).item()
                degrees = preds.sum(dim=1)
                preds = torch.diag(1./degrees)@preds
                mod_pred = make_modularity_matrix(preds)
                r = baseline_spectral(mod_pred, K)
                vals['train-spectral'][idx] = loss_fn(None, r, None, None, bin_adj_all[i], test_object[i], args).item()
            elif args.objective == 'kcenter':
                loss_gonzalez = 0
                loss_greedy = 0
                for i in test_instances:
                    x = gonzalez_kcenter(dist_train[i], K)
                    vals['train-gonzalez'][idx] = test_object[i](x).item()
                    x = greedy_kcenter(dist_train[i], diameter, K)
                    vals['train-greedy'][idx] = test_object[i](x).item()
        for algo in algs:
            if 'train' in algo:
                print(algo, np.mean(vals[algo]), np.std(vals[algo]))

        
        
print()
for algo in algs:
    print(algo, np.mean(vals[algo]), np.std(vals[algo]))

pickle.dump((vals, aucs), open('results_distributional_{}_{}_{}.pickle'.format(args.dataset, args.objective, args.K), 'wb'))