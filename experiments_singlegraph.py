from pygcn import load_data
import torch
import argparse
import numpy as np
import torch.optim as optim
import torch.nn as nn
import sklearn
from kcenter import make_all_dists, greedy_kcenter, gonzalez_kcenter, CenterObjective, make_dists_igraph, rounding
from models import GCNLink, GCNClusterNet, GCNDeep, GCNDeepSigmoid, GCN
from utils import make_normalized_adj, negative_sample, edge_dropout, load_nofeatures
from modularity import baseline_spectral, partition, greedy_modularity_communities, make_modularity_matrix
from loss_functions import loss_kcenter, loss_modularity


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
parser.add_argument('--objective', type=str, default='modularity',
                    help='What objective to optimize (currently kenter or modularity)')
parser.add_argument('--dataset', type=str, default='citeseer',
                    help='which network to load')
parser.add_argument('--clustertemp', type=float, default=30,
                    help='how hard to make the softmax for the cluster assignments')
parser.add_argument('--kcentertemp', type=float, default=100,
                    help='how hard to make seed selection softmax assignment')
parser.add_argument('--kcentermintemp', type=float, default=0,
                    help='how hard to make the min over nodes in kcenter training objective')
parser.add_argument('--train_pct', type=float, default=0.4, help='percent of total edges in training set')
parser.add_argument('--calculate_opt', action='store_true', default=False, help='calculate opt')
parser.add_argument('--pure_opt', action='store_true', default=False, help='do only optimization, no link prediction needed')
parser.add_argument('--use_igraph', action='store_true', default=True, help='use igraph to compute shortest paths in twostage kcenter')
parser.add_argument('--run_ts', action='store_true', default=False, help='do only optimization, no link prediction needed')
parser.add_argument('--train_iters', type=int, default=1001,
                    help='number of training iterations')
parser.add_argument('--num_cluster_iter', type=int, default=1,
                    help='number of iterations for clustering')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

pure_opt = args.pure_opt

reload_data = True

test_cluster_auc = False

calculate_opt = args.calculate_opt

make_objectives = False
if reload_data:
    make_objectives = True

calculate_dists = False

run_decision = True
run_ts = args.run_ts
run_gcne2e = True
run_train_only = True

has_features = True

##############################################################################
#LOAD DATA
##############################################################################
train_pct = args.train_pct

if reload_data:
    if has_features:
        adj_test, features_test, labels, idx_train, idx_val, idx_test = load_data('data/{}/'.format(args.dataset), '{}_test_{:.2f}'.format(args.dataset, train_pct))
        adj_valid, features_valid, labels, idx_train, idx_val, idx_test = load_data('data/{}/'.format(args.dataset), '{}_valid_{:.2f}'.format(args.dataset, train_pct))
        adj_train, features_train, labels, idx_train, idx_val, idx_test = load_data('data/{}/'.format(args.dataset), '{}_train_{:.2f}'.format(args.dataset, train_pct))
    else:
        adj_all, features, labels = load_nofeatures(args.dataset, '')
        features_train = features
        features_test = features
        n = adj_all.shape[0]
        adj_train, features, labels = load_nofeatures(args.dataset, '_train_{:.2f}'.format(train_pct), n)
        adj_test, features, labels = load_nofeatures(args.dataset, '_test_{:.2f}'.format(train_pct), n)
        adj_valid, features, labels = load_nofeatures(args.dataset, '_valid_{:.2f}'.format(train_pct), n)


adj_test = adj_test.coalesce()
adj_valid = adj_valid.coalesce()
adj_train = adj_train.coalesce()
n = adj_train.shape[0]
K = args.K
bin_adj_test = (adj_test.to_dense() > 0).float()
bin_adj_train = (adj_train.to_dense() > 0).float()
m_train = bin_adj_train.sum()
bin_adj_valid = (adj_valid.to_dense() > 0).float()
bin_adj_all = (bin_adj_train + bin_adj_test + bin_adj_valid > 0).float()
adj_all = make_normalized_adj(bin_adj_all.nonzero(), n)
nfeat = features_test.shape[1]

adj_all, features_test, labels, idx_train, idx_val, idx_test = load_data('data/{}/'.format(args.dataset), '{}'.format(args.dataset))
adj_all = adj_all.coalesce()
adj_test = adj_all
bin_adj_all = (adj_all.to_dense() > 0).float()
n = adj_all.shape[0]
K= args.K
nfeat = features_test.shape[1]

##############################################################################
#INITIALIZE MODELS
##############################################################################

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

#uses GCNs to predict the cluster membership of each node
model_gcn = GCNDeep(nfeat=nfeat,
            nhid=args.hidden,
            nout=args.K,
            dropout=args.dropout,
            nlayers=2)

#uses GCNs to predict the probability that each node appears in the solution
model_gcn_x = GCNDeepSigmoid(nfeat=nfeat,
            nhid=args.hidden,
            nout=1,
            dropout=args.dropout,
            nlayers=2)

if args.objective == 'kcenter':
    model_gcn = model_gcn_x



optimizer = optim.Adam(model_cluster.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model_cluster.cuda()
    model_ts.cuda()
    features = features.cuda()
    adj_train = adj_train.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

losses = []
losses_test = []
num_cluster_iter = args.num_cluster_iter

##############################################################################
#MAKE AUXILIARY DATA FOR OBJECTIVES
##############################################################################

if make_objectives:
    if args.objective == 'kcenter':
        try:
            dist_all = torch.load('{}_test_dist.pt'.format(args.dataset))
            dist_train = torch.load('{}_{}_train_dist.pt'.format(args.dataset, train_pct))
            diameter = dist_all.max()
        except:
            dist_all = make_all_dists(bin_adj_all, 100)
            diameter = dist_all[dist_all < 100].max()
            dist_all[dist_all == 100] = diameter
            torch.save(dist_all, '{}_test_dist.pt'.format(args.dataset))
            dist_train = make_all_dists(bin_adj_train, 100)
            dist_train[dist_train == 100] = diameter
            torch.save(dist_train, '{}_{}_train_dist.pt'.format(args.dataset, train_pct))
        obj_train = CenterObjective(dist_train, diameter, args.kcentermintemp)
        obj_train_hardmax = CenterObjective(dist_train, diameter, args.kcentermintemp, hardmax=True)
        obj_test = CenterObjective(dist_all, diameter, args.kcentertemp, hardmax=True)
        obj_test_softmax = CenterObjective(dist_all, diameter, args.kcentermintemp)

    if args.objective == 'modularity':
        mod_train = make_modularity_matrix(bin_adj_train)
        mod_test = make_modularity_matrix(bin_adj_test)
        mod_valid = make_modularity_matrix(bin_adj_valid)
        mod_all = make_modularity_matrix(bin_adj_all)

##############################################################################
#DEFINE LOSS FUNCTIONS
##############################################################################


if args.objective == 'modularity':
    loss_fn = loss_modularity
    test_object = mod_all
    train_object = mod_train
    test_only_object = mod_test
    valid_object = mod_valid
elif args.objective == 'kcenter':
    loss_fn = loss_kcenter
    test_object= obj_test
    train_object = obj_train
    test_only_object = None
    valid_object = None
else:
    raise Exception('unknown objective')


##############################################################################
#TRAIN DECISION-FOCUSED
##############################################################################

#Decision-focused training
best_train_val = 100
if run_decision:
    for t in range(args.train_iters):
        #pure optimization setting: get loss with respect to the full graph
        if pure_opt:
            mu, r, embeds, dist = model_cluster(features_test, adj_all, num_cluster_iter)
            loss = loss_fn(mu, r, embeds, dist, bin_adj_all, test_object, args)
        #link prediction setting: get loss with respect to training edges only
        else:
            mu, r, embeds, dist = model_cluster(features_train, adj_train, num_cluster_iter)
            loss = loss_fn(mu, r, embeds, dist, bin_adj_train, train_object, args)
        if args.objective != 'kcenter':
            loss = -loss
        optimizer.zero_grad()
        loss.backward()
        #increase number of clustering iterations after 500 updates to fine-tune
        #solution
        if t == 500:
            num_cluster_iter = 5
        #every 100 iterations, look and see if we've improved on the best training loss
        #seen so far. Keep the solution with best training value.
        if t % 100 == 0:
            #round solution to discrete partitioning
            if args.objective == 'modularity':
                r = torch.softmax(100*r, dim=1)
            #evalaute test loss -- note that the best solution is
            #chosen with respect training loss. Here, we store the test loss
            #of the currently best training solution
            loss_test = loss_fn(mu, r, embeds, dist, bin_adj_all, test_object, args)
            #for k-center problem, keep track of the fractional x with best
            #training loss, to do rounding after
            if loss.item() < best_train_val:
                best_train_val = loss.item()
                curr_test_loss = loss_test.item()
                #convert distances into a feasible (fractional x)
                x_best = torch.softmax(dist*args.kcentertemp, 0).sum(dim=1)
                x_best = 2*(torch.sigmoid(4*x_best) - 0.5)
                if x_best.sum() > K:
                    x_best = K*x_best/x_best.sum()
        losses.append(loss.item())
        optimizer.step()

    #for k-center: round 50 times and take the solution with best training
    #value
    if args.objective == 'kcenter':
        testvals = []; trainvals = []
        for _ in range(50):
            y = rounding(x_best)
            testvals.append(obj_test(y).item())
            trainvals.append(obj_train(y).item())
        print('ClusterNet value', testvals[np.argmin(trainvals)])
    if args.objective == 'modularity':
            print('ClusterNet value', curr_test_loss)

##############################################################################
#TRAIN TWO-STAGE
##############################################################################

def train_twostage(model_ts):
    optimizer_ts = optim.Adam(model_ts.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
    edges = adj_train.indices().t()
    edges_test = adj_test.indices().t()
    edges_test_eval, labels_test_eval = negative_sample(edges_test, 1, bin_adj_train)
#    print(edges_test_eval)
    for t in range(300):
        adj_input = make_normalized_adj(edge_dropout(edges, args.edge_dropout), n)
        edges_eval, labels = negative_sample(edges, args.negsamplerate, bin_adj_train)
        preds = model_ts(features_train, adj_input, edges_eval)
        loss = torch.nn.BCEWithLogitsLoss()(preds, labels)
        optimizer_ts.zero_grad()
        loss.backward()
        if t % 10 == 0:
            preds_test_eval = model_ts(features_train, adj_input, edges_test_eval)
            test_ce = torch.nn.BCEWithLogitsLoss()(preds_test_eval, labels_test_eval)
            test_auc = sklearn.metrics.roc_auc_score(labels_test_eval.long().detach().numpy(), nn.Sigmoid()(preds_test_eval).detach().numpy())
            print(t, loss.item(), test_ce.item(), test_auc)
        optimizer_ts.step()

if test_cluster_auc:
    model_linkpred = GCNLink(nfeat=nfeat,
            nhid=args.hidden,
            nout=args.embed_dim,
            dropout=args.dropout)
    model_linkpred.GCN = model_cluster.GCN
    model_linkpred.GCN.requires_grad = False
    train_twostage(model_linkpred)


calculate_ts_performance = False
if run_ts:
    print('two stage')
    train_twostage(model_ts)
    #predict probability that all unobserved edges exist
    indices = torch.tensor(np.arange(n))
    to_pred = torch.zeros(n**2, 2)
    to_pred[:, 1] = indices.repeat(n)
    for i in range(n):
        to_pred[i*n:(i+1)*n, 0] = i
    to_pred = to_pred.long()
    preds = model_ts(features_train, adj_train, to_pred)
    preds = nn.Sigmoid()(preds).view(n, n)

    preds = bin_adj_train + (1 - bin_adj_train)*preds

    if args.objective == 'modularity':
        r = greedy_modularity_communities(preds, K)
        print('agglomerative', loss_fn(None, r, None, None, bin_adj_all, test_object, args))
        r = partition(preds, K)
        print('recursive', loss_fn(None, r, None, None, bin_adj_all, test_object, args))
        degrees = preds.sum(dim=1)
        preds = torch.diag(1./degrees)@preds
        mod_pred = make_modularity_matrix(preds)
        r = baseline_spectral(mod_pred, K)
        print('spectral', loss_fn(None, r, None, None, bin_adj_all, test_object, args))
    elif args.objective == 'kcenter':
        try:
            dist_ts = torch.load('{}_twostage_dist.pt'.format(args.dataset))
            print('loaded ts dists from {}'.format('{}_twostage_dist.pt'.format(args.dataset)))
        except:
            print('making dists')
            if args.use_igraph:
                print('using igraph')
                dist_ts =  make_dists_igraph(preds)
            else:
                print('using networkx')
                dist_ts = make_all_dists(preds, 100)
                diameter = dist_ts[dist_ts < 100].max()
                dist_ts[dist_ts == 100] = diameter
            print('made dists')
            torch.save(dist_ts, '{}_twostage_dist.pt'.format(args.dataset))
        dist_ts = dist_ts.float()
        diameter = dist_ts.max()
        x = gonzalez_kcenter(dist_ts, K)
        print('gonzalez ts', obj_train_hardmax(x), obj_test(x))
        print(dist_ts.type(), diameter.type())
        x = greedy_kcenter(dist_ts, diameter, K)
        print('greedy ts', obj_train_hardmax(x), obj_test(x))

##############################################################################
#TRAIN END-TO-END GCN
##############################################################################

if run_gcne2e:
    print('just GCN')
    optimizer_gcn = optim.Adam(model_gcn.parameters(), lr = args.lr,
                               weight_decay = args.weight_decay)
    if args.objective == 'modularity':
        best_train_val = 0
    if args.objective == 'kcenter':
        best_train_val = 100

    for t in range(1000):
        best_train_loss = 100
        if pure_opt:
            if args.objective == 'modularity' or args.objective == 'maxcut':
                r = model_gcn(features_test, adj_all)
                r = torch.softmax(args.clustertemp*r, dim = 1)
                loss = -loss_fn(None, r, None, None, bin_adj_train, train_object, args)
            elif args.objective == 'kcenter' or args.objecive == 'influmax':
                x = model_gcn(features_test, adj_all)
                if x.sum() > K:
                    x = K*x/x.sum()
                loss = -test_object(x)
        else:
            if args.objective == 'modularity' or args.objective == 'maxcut':
                r = model_gcn(features_train, adj_train)
                r = torch.softmax(r, dim = 1)
                loss = -loss_fn(None, r, None, None, bin_adj_train, train_object, args)
            elif args.objective == 'kcenter' or args.objecive == 'influmax':
                x = model_gcn(features_train, adj_train)
                if x.sum() > K:
                    x = K*x/x.sum()
                loss = -train_object(x)
        if args.objective == 'kcenter':
            loss = -loss
        optimizer.zero_grad()
        loss.backward()
        if t % 100 == 0:
            if args.objective == 'modularity' or args.objective == 'maxcut':
                r = torch.softmax(100*r, dim=1)
                loss_test = loss_fn(None, r, None, None, bin_adj_all, test_object, args)
                loss_test_only = loss_fn(None, r, None, None, bin_adj_test, test_only_object, args)
            elif args.objective == 'kcenter' or args.objecive == 'influmax':
                loss_test = -test_object(x)
                loss_test_only = torch.tensor(0).float()
            losses_test.append(loss_test.item())
            print(t, loss.item(), loss_test.item(), loss_test_only.item())
            if loss.item() < best_train_val:
                curr_test_loss = loss_test.item()
                best_train_val = loss.item()
                if args.objective == 'kcenter' or args.objective == 'influmax':
                    x_best = x
        losses.append(loss.item())
        optimizer.step()
    if args.objective == 'kcenter':
        from influmax import rounding
        testvals = []; trainvals = []; trainvalshardmax = []
        for _ in range(50):
            y = rounding(x_best)
            testvals.append(obj_test(y).item())
            trainvals.append(obj_train(y).item())
            trainvalshardmax.append(obj_train_hardmax(y).item())
        print('train min', testvals[np.argmin(trainvals)])
        print('hardmax train min', testvals[np.argmin(trainvalshardmax)])
        print('absolute min', min(testvals))
    if args.objective == 'modularity':
        print('train min', curr_test_loss)


##############################################################################
#TRAIN-ONLY BASELINE
##############################################################################

if run_train_only:
    if args.objective == 'modularity':
        preds = bin_adj_train
        r = greedy_modularity_communities(preds, K)
        print('agglomerative', loss_fn(None, r, None, None, bin_adj_all, test_object, args))
        r = partition(preds, K)
        print('recursive', loss_fn(None, r, None, None, bin_adj_all, test_object, args))
        degrees = preds.sum(dim=1)
        preds = torch.diag(1./degrees)@preds
        mod_pred = make_modularity_matrix(preds)
        r = baseline_spectral(mod_pred, K)
        print('spectral', loss_fn(None, r, None, None, bin_adj_all, test_object, args))
    elif args.objective == 'kcenter':
        x = gonzalez_kcenter(dist_train, K)
        print('gonzalez train', obj_test(x))
        x = greedy_kcenter(dist_train, diameter, K)
        print('greedy train', obj_test(x))


##############################################################################
#RUN BASELINE OPTIMIZATION ALGORITHMS ON FULL GRAPH
##############################################################################

if calculate_opt:
    if args.objective == 'modularity':
        preds = bin_adj_all
        r = greedy_modularity_communities(preds, K)
        print('agglomerative', loss_fn(None, r, None, None, bin_adj_all, test_object, args))
        r = partition(preds, K)
        print('recursive', loss_fn(None, r, None, None, bin_adj_all, test_object, args))
        degrees = preds.sum(dim=1)
        preds = torch.diag(1./degrees)@preds
        mod_pred = make_modularity_matrix(preds)
        r = baseline_spectral(mod_pred, K)
        print('spectral', loss_fn(None, r, None, None, bin_adj_all, test_object, args))
    elif args.objective == 'kcenter':
        x = gonzalez_kcenter(dist_all, K)
        print('gonzalez all', obj_test(x))
        x = greedy_kcenter(dist_all, diameter, K)
        print('greedy all', obj_test(x))