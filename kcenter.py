import torch
import numpy as np
import networkx as nx

class CenterObjective():
    def __init__(self, dist, dmax, temp, hardmax=False):
        '''
        dist: (num customers) * (num locations) matrix
        
        dmax: maximum distance that can be suffered by any customer (e.g., if 
              no facilities are chosen)
        
        temp: how hard to make the softmax over customers
        '''
        self.dmax = dmax
        dist, order = torch.sort(dist, dim=1)
        self.order = order
        dmax_vec = dmax*torch.ones(dist.shape[0], 1)
        off_one = torch.cat((dist[:, 1:], dmax_vec), dim=1)
        self.m = dist - off_one
        self.temp = temp
        self.hardmax = hardmax

    def __call__(self, x):
        '''
        Evaluates E_S[softmax_{customers} min_{i \in S} dist(customer, i)] where 
        the expectation is over the set of facility locations S. Every 
        location is included in S independently with probability x_i. 
        '''
        x_sort = x[self.order]
        probs = 1 - torch.cumprod(1 - x_sort, dim=1)
        vals = self.dmax + (self.m*probs).sum(dim=1)
        if self.hardmax:
            return vals.max()
        weights = torch.softmax(self.temp*vals, dim=0)
        return torch.dot(vals, weights)


def gonzalez_kcenter(dist, K):
    '''
    Algorithm of Gonzalez (1985) which iteratively selects the point furthest
    from the current solution
    
    Gonzalez, Teofilo F. (1985). "Clustering to minimize the maximum intercluster 
    distance". Theoretical Computer Science.
    '''
    S = [np.random.choice(list(range(dist.shape[1])))]
    while len(S) < K:
        dist_to_S = dist[:, S].min(dim = 1)[0]
        S.append(dist_to_S.argmax().item())
    x = torch.zeros(dist.shape[1])
    x[S] = 1
    return x

def greedy_kcenter(dist, dmax, K):
    '''
    Greedily add locations to minimize the kcenter objective
    '''
    obj = CenterObjective(dist, dmax, None, True)
    x = torch.zeros(dist.shape[1])
    currval = obj(x)
    for _ in range(K):
        best_i = 0
        for i in range(dist.shape[1]):
             if x[i] < 0.5:
                 x[i] = 1
                 obj_val = obj(x)
                 if obj_val < currval:
                     currval = obj_val
                     best_i = i
                 x[i] = 0
        x[best_i] = 1
    return x

def make_all_dists(bin_adj, dmax, use_weights=False):
    g = nx.from_numpy_array(bin_adj.detach().numpy())
    if not use_weights:
        lengths = nx.shortest_path_length(g)
    else:
        lengths = nx.shortest_path_length(g, weight='weight')
    dist = torch.zeros_like(bin_adj)
    for u, lens_u in lengths:
        for v in range(bin_adj.shape[0]):
            if v in lens_u:
                dist[u,v] = lens_u[v]
            else:
                dist[u,v] = dmax
    return dist

def make_dists_igraph(adj):
    import igraph
    adj = adj.detach().numpy()
    dense = np.random.rand(adj.shape[0], adj.shape[1])
    e1 = dense.nonzero()[0]
    e1 = e1.reshape(e1.shape[0], 1)
    e2 = dense.nonzero()[1]
    e2 = e2.reshape(e2.shape[0], 1)
    stuff = np.concatenate((e1, e2), axis=1)
    allstuff = np.concatenate((stuff, adj.flatten().reshape(stuff.shape[0], 1)), axis=1)
    np.savetxt('tmp_twostage', allstuff, fmt = '%d %d %f')
    g = igraph.Graph.Read_Ncol('tmp_twostage', weights=True, directed=True)
    dists = g.shortest_paths(weights='weight')
    dists = torch.tensor(np.array(dists))
    return dists.float()

def rounding(x):
    '''
    Fast pipage rounding implementation for uniform matroid
    '''
    i = 0
    j = 1
    x = x.clone()
    for t in range(len(x)-1):
        if x[i] == 0 and x[j] == 0:
            i = max((i,j)) + 1
        elif x[i] + x[j] < 1:
            if np.random.rand() < x[i]/(x[i] + x[j]):
                x[i] = x[i] + x[j]
                x[j] = 0
                j = max((i,j)) + 1
            else:
                x[j] = x[i] + x[j]
                x[i] = 0
                i = max((i,j)) + 1
        else:
            if np.random.rand() < (1 - x[j])/(2 - x[i] - x[j]):
                x[j] = x[i] + x[j] - 1
                x[i] = 1
                i = max((i,j)) + 1

            else:
                x[i] = x[i] + x[j] - 1
                x[j] = 1
                j = max((i,j)) + 1
    return x
