import numpy as np
import networkx as nx
from scipy import sparse
from scipy.linalg import eig
from itertools import product
import torch
from networkx.algorithms.community.quality import modularity
from networkx.utils.mapped_queue import MappedQueue
import heapq
import scipy as sp
import scipy.linalg

def make_modularity_matrix(adj):
    adj = adj*(torch.ones(adj.shape[0], adj.shape[0]) - torch.eye(adj.shape[0]))
    degrees = adj.sum(dim=0).unsqueeze(1)
    mod = adj - degrees@degrees.t()/adj.sum()
    return mod

def make_modularity_matrix_nodiag(adj):
    adj = adj*(torch.ones(adj.shape[0], adj.shape[0]) - torch.eye(adj.shape[0]))
    degrees = adj.sum(dim=0).unsqueeze(1)
    mod = adj - degrees@degrees.t()/adj.sum()
    return mod

def baseline_spectral(mod, K):
    import sklearn
    B = mod.detach().numpy()
    w, v = sp.linalg.eigh(B, eigvals=(B.shape[0] - 1 - K, B.shape[0]-1))
    part = sklearn.cluster.k_means(v, K)[1]
    r = torch.zeros(B.shape[0], K)
    for i in range(mod.shape[0]):
        r[i, part[i]] = 1
    return r


#  Code modified from https://networkx.github.io/documentation/latest/_modules/networkx/algorithms/community/modularity_max.html#greedy_modularity_communities
#  Added changes to handle weighted networks and terminate with a given number of
#  communities

def greedy_modularity_communities(G, K, weight=None):
    """Find communities in graph using Clauset-Newman-Moore greedy modularity
    maximization. This method currently supports the Graph class and does not
    consider edge weights.

    Greedy modularity maximization begins with each node in its own community
    and joins the pair of communities that most increases modularity until no
    such pair exists.
    
    Parameters
    ----------
    G : NetworkX graph

    Returns
    -------
    Yields sets of nodes, one for each community.

    Examples
    --------
    >>> from networkx.algorithms.community import greedy_modularity_communities
    >>> G = nx.karate_club_graph()
    >>> c = list(greedy_modularity_communities(G))
    >>> sorted(c[0])
    [8, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

    References
    ----------
    .. [1] M. E. J Newman 'Networks: An Introduction', page 224
       Oxford University Press 2011.
    .. [2] Clauset, A., Newman, M. E., & Moore, C.
       "Finding community structure in very large networks."
       Physical Review E 70(6), 2004.
    """
    G = nx.from_numpy_array(G.detach().numpy(), create_using = nx.DiGraph())
    # Count nodes and edges
    N = len(G.nodes())
    m = sum([d.get('weight', 1) for u, v, d in G.edges(data=True)])
    q0 = 1.0 / (2.0*m)

    # Map node labels to contiguous integers
    label_for_node = dict((i, v) for i, v in enumerate(G.nodes()))
    node_for_label = dict((label_for_node[i], i) for i in range(N))

    # Calculate degrees
    k_for_label = G.degree(G.nodes(), weight=weight)
    k = [k_for_label[label_for_node[i]] for i in range(N)]

    # Initialize community and merge lists
    communities = dict((i, frozenset([i])) for i in range(N))
    merges = []

    # Initial modularity
    partition = [[label_for_node[x] for x in c] for c in communities.values()]
    q_cnm = modularity(G, partition)

    # Initialize data structures
    # CNM Eq 8-9 (Eq 8 was missing a factor of 2 (from A_ij + A_ji)
    # a[i]: fraction of edges within community i
    # dq_dict[i][j]: dQ for merging community i, j
    # dq_heap[i][n] : (-dq, i, j) for communitiy i nth largest dQ
    # H[n]: (-dq, i, j) for community with nth largest max_j(dQ_ij)
    a = [k[i]*q0 for i in range(N)]
    dq_dict = dict(
        (i, dict(
            (j, q0*(G[i][j]['weight'] + G[j][i]['weight']) - 2*k[i]*k[j]*q0*q0)
            for j in [
                node_for_label[u]
                for u in G.neighbors(label_for_node[i])]
            if j != i))
        for i in range(N))
#    print(min([len(x[1]) for x in dq_dict.values()]))
#    raise Exception()
#    return dq_dict
    dq_heap = [
        MappedQueue([
            (-dq, i, j)
            for j, dq in dq_dict[i].items()])
        for i in range(N)]
    H = MappedQueue([
        dq_heap[i].h[0]
        for i in range(N)
        if len(dq_heap[i]) > 0])

    # Merge communities until we can't improve modularity
    while len(H) > 1:
        # Find best merge
        # Remove from heap of row maxes
        # Ties will be broken by choosing the pair with lowest min community id
        try:
            dq, i, j = H.pop()
        except IndexError:
            break
        dq = -dq
        # Remove best merge from row i heap
        dq_heap[i].pop()
        # Push new row max onto H
        if len(dq_heap[i]) > 0:
            H.push(dq_heap[i].h[0])
        # If this element was also at the root of row j, we need to remove the
        # duplicate entry from H
        if dq_heap[j].h[0] == (-dq, j, i):
            H.remove((-dq, j, i))
            # Remove best merge from row j heap
            dq_heap[j].remove((-dq, j, i))
            # Push new row max onto H
            if len(dq_heap[j]) > 0:
                H.push(dq_heap[j].h[0])
        else:
            # Duplicate wasn't in H, just remove from row j heap
            dq_heap[j].remove((-dq, j, i))

        # Perform merge
        communities[j] = frozenset(communities[i] | communities[j])
        del communities[i]
        merges.append((i, j, dq))
        
#        print(len(communities))
        # Stop when change is non-positive
        if len(communities) == K:
            break

        # New modularity
        q_cnm += dq
        # Get list of communities connected to merged communities
        i_set = set(dq_dict[i].keys())
        j_set = set(dq_dict[j].keys())
        all_set = (i_set | j_set) - set([i, j])
        both_set = i_set & j_set
        # Merge i into j and update dQ
        for k in all_set:
            # Calculate new dq value
            if k in both_set:
                dq_jk = dq_dict[j][k] + dq_dict[i][k]
            elif k in j_set:
                dq_jk = dq_dict[j][k] - 2.0*a[i]*a[k]
            else:
                # k in i_set
                dq_jk = dq_dict[i][k] - 2.0*a[j]*a[k]
            # Update rows j and k
            for row, col in [(j, k), (k, j)]:
                # Save old value for finding heap index
                if k in j_set:
                    d_old = (-dq_dict[row][col], row, col)
                else:
                    d_old = None
                # Update dict for j,k only (i is removed below)
                dq_dict[row][col] = dq_jk
                # Save old max of per-row heap
                if len(dq_heap[row]) > 0:
                    d_oldmax = dq_heap[row].h[0]
                else:
                    d_oldmax = None
                # Add/update heaps
                d = (-dq_jk, row, col)
                if d_old is None:
                    # We're creating a new nonzero element, add to heap
                    dq_heap[row].push(d)
                else:
                    # Update existing element in per-row heap
                    dq_heap[row].update(d_old, d)
                # Update heap of row maxes if necessary
                if d_oldmax is None:
                    # No entries previously in this row, push new max
                    H.push(d)
                else:
                    # We've updated an entry in this row, has the max changed?
                    if dq_heap[row].h[0] != d_oldmax:
                        H.update(d_oldmax, dq_heap[row].h[0])

        # Remove row/col i from matrix
        i_neighbors = dq_dict[i].keys()
        for k in i_neighbors:
            # Remove from dict
            dq_old = dq_dict[k][i]
            del dq_dict[k][i]
            # Remove from heaps if we haven't already
            if k != j:
                # Remove both row and column
                for row, col in [(k, i), (i, k)]:
                    # Check if replaced dq is row max
                    d_old = (-dq_old, row, col)
                    if dq_heap[row].h[0] == d_old:
                        # Update per-row heap and heap of row maxes
                        dq_heap[row].remove(d_old)
                        H.remove(d_old)
                        # Update row max
                        if len(dq_heap[row]) > 0:
                            H.push(dq_heap[row].h[0])
                    else:
                        # Only update per-row heap
#                        if d_old in dq_heap[row].d:
                        dq_heap[row].remove(d_old)

        del dq_dict[i]
        # Mark row i as deleted, but keep placeholder
        dq_heap[i] = MappedQueue()
        # Merge i into j and update a
        a[j] += a[i]
        a[i] = 0

#    communities = [
#        set([label_for_node[i] for i in c])
#        for c in communities.values()]
    heap = []
    for j in communities:
        heapq.heappush(heap, (a[j], set(communities[j])))
    while len(heap) > K:
        weight1, com1 = heapq.heappop(heap)
        weight2, com2 = heapq.heappop(heap)
        com1.update(com2)
        heapq.heappush(heap, (weight1+weight2, com1))
    communities = [x[1] for x in heap]
    r = torch.zeros(N, K)
    print(len(communities))
    for i,c in enumerate(communities):
        for v in c:
            r[v, i] = 1
    return r


'''
Code modified from https://github.com/zhiyzuo/python-modularity-maximization
'''

def partition(adj, K, refine=False):
    '''
    Cluster a network into several modules
    using modularity maximization by spectral methods.
    Supports directed and undirected networks.
    Edge weights are ignored
    See:
    Newman, M. E. J. (2006). Modularity and community structure in networks.
    Proceedings of the National Academy of Sciences of the United States of America,
    103(23), 8577–82. https://doi.org/10.1073/pnas.0601602103
    Leicht, E. A., & Newman, M. E. J. (2008). Community Structure in Directed Networks.
    Physical Review Letters, 100(11), 118703. https://doi.org/10.1103/PhysRevLett.100.118703
    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        The network of interest
    refine: Boolean
        Whether refine the `s` vector from the initial clustering
        by repeatedly moving nodes to maximize modularity
    Returns
    -------
    dict
        A dictionary that saves membership.
        Key: node label; Value: community index
    '''
    ## preprocessing
    network = nx.from_numpy_array(adj.detach().numpy(), create_using = nx.DiGraph())
#    network = nx.convert_node_labels_to_integers(network, first_label=1, label_attribute="node_name")

    B = get_base_modularity_matrix(network)

    ## set flags for divisibility of communities
    ## initial community is divisible
#    divisible_community = deque([0])
    divisible_community = [0]

    ## add attributes: all node as one group
    community_dict = {u: 0 for u in network}

    ## overall modularity matrix

    comm_counter = 1

#    while len(divisible_community) > 0:
    while comm_counter < K:
        best_delta = -np.inf
        best_index = None
        for comm_index in divisible_community:
            ## get the first divisible comm index out
            g1_nodes, comm_nodes, delta_mod = _divide(network, community_dict, comm_index, B, refine)
            if delta_mod > best_delta and g1_nodes is not None:
                best_delta = delta_mod
                best_index = comm_index
          
        if best_index is None:
            raise Exception('no division into K groups')
        comm_index = best_index
        g1_nodes, comm_nodes, delta_mod = _divide(network, community_dict, comm_index, B, refine)
        if g1_nodes is None:
            raise Exception('no division into K groups')
        g2 = network.subgraph(set(comm_nodes).difference(set(g1_nodes)))
        for u in g2:
            community_dict[u] = comm_counter
        divisible_community.append(comm_counter)
        comm_counter += 1
    r = torch.zeros(B.shape[0], K)
    for i in range(B.shape[0]):
        r[i, community_dict[i]] = 1
    return r

def _divide(network, community_dict, comm_index, B, refine=False):
    '''
    Bisection of a community in `network`.
    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        The network of interest
    Returns
    -------
    tuple
        If the given community is indivisible, return (None, None)
        If the given community is divisible, return a tuple where
        the 1st element is a node list for the 1st sub-group and
        the 2nd element is a node list for the original group
    '''

    comm_nodes = tuple(u for u in community_dict \
                  if community_dict[u] == comm_index)
    B_hat_g = get_mod_matrix(network, comm_nodes, B)

    # compute the top eigenvector u₁ and β₁
    if B_hat_g.shape[0] < 3:
        beta_s, u_s = largest_eig(B_hat_g)
    else:
        beta_s, u_s = sparse.linalg.eigs(B_hat_g, k=1, which='LR')
    u_1 = u_s[:, 0]
    beta_1 = beta_s[0]
#    if beta_1 > 0:
        # divisible
    s = sparse.csc_matrix(np.asmatrix([[1 if u_1_i > 0 else -1] for u_1_i in u_1]))
    if refine:
        improve_modularity(network, comm_nodes, s, B)
    delta_modularity = _get_delta_Q(B_hat_g, s)
    g1_nodes = np.array([comm_nodes[i] \
                         for i in range(u_1.shape[0]) \
                         if s[i,0] > 0])
    #g1 = nx.subgraph(g, g1_nodes)
    if len(g1_nodes) == len(comm_nodes) or len(g1_nodes) == 0:
        # indivisble, return None
        return None, None, delta_modularity
    # divisible, return node list for one of the groups
    return g1_nodes, comm_nodes, delta_modularity
    # indivisble, return None
#    return None, None, beta_1

def improve_modularity(network, comm_nodes, s, B):
    '''
    Fine tuning of the initial division from `_divide`
    Modify `s` inplace
    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        The network of interest
    comm_nodes: iterable
        List of nodes for the original group
    s: np.matrix
        A matrix of node membership. Only +1/-1
    B: np.amtrix
        Modularity matrix for `network`
    '''

    # iterate until no increment of Q
    B_hat_g = get_mod_matrix(network, comm_nodes, B)
    while True:
        unmoved = list(comm_nodes)
        # node indices to be moved
        node_indices = np.array([], dtype=int)
        # cumulative improvement after moving
        node_improvement = np.array([], dtype=float)
        # keep moving until none left
        while len(unmoved) > 0:
            # init Q
            Q0 = _get_delta_Q(B_hat_g, s)
            scores = np.zeros(len(unmoved))
            for k_index in range(scores.size):
                k = comm_nodes.index(unmoved[k_index])
                s[k, 0] = -s[k, 0]
                scores[k_index] = _get_delta_Q(B_hat_g, s) - Q0
                s[k, 0] = -s[k, 0]
            _j = np.argmax(scores)
            j = comm_nodes.index(unmoved[_j])
            # move j, which has the largest increase or smallest decrease
            s[j, 0] = -s[j, 0]
            node_indices = np.append(node_indices, j)
            if node_improvement.size < 1:
                node_improvement = np.append(node_improvement, scores[_j])
            else:
                node_improvement = np.append(node_improvement, \
                                        node_improvement[-1]+scores[_j])
            #print len(unmoved), 'max: ', max(scores), node_improvement[-1]
            unmoved.pop(_j)
        # the biggest improvement
        max_index = np.argmax(node_improvement)
        # change all the remaining nodes
        # which are not helping
        for i in range(max_index+1, len(comm_nodes)):
            j = node_indices[i]
            s[j,0] = -s[j, 0]
        # if we swap all the nodes, it is actually doing nothing
        if max_index == len(comm_nodes) - 1:
            delta_modularity = 0
        else:
            delta_modularity = node_improvement[max_index]
        # Stop if ΔQ <= 0 
        if delta_modularity <= 0:
            break


def get_base_modularity_matrix(network):
    '''
    Obtain the modularity matrix for the whole network
    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        The network of interest
    Returns
    -------
    np.matrix
        The modularity matrix for `network`
    Raises
    ------
    TypeError
        When the input `network` does not fit either nx.Graph or nx.DiGraph
    '''

    if type(network) == nx.Graph:
        return sparse.csc_matrix(nx.modularity_matrix(network))
    elif type(network) == nx.DiGraph:
        return sparse.csc_matrix(nx.directed_modularity_matrix(network))
    else:
        raise TypeError('Graph type not supported. Use either nx.Graph or nx.Digraph')

def _get_delta_Q(X, a):
    '''
    Calculate the detal modularity
    .. math::
        \deltaQ = s^T \cdot \^{B_{g}} \cdot s
    .. math:: \deltaQ = s^T \cdot \^{B_{g}} \cdot s
    Parameters
    ----------
    X : np.matrix
        B_hat_g
    a : np.matrix
        s, which is the membership vector
    Returns
    -------
    float
        The corresponding :math:`\deltaQ`
    '''

    delta_Q = (a.T.dot(X)).dot(a)

    return delta_Q[0,0]

def get_modularity(network, community_dict):
    '''
    Calculate the modularity. Edge weights are ignored.
    Undirected:
    .. math:: Q = \frac{1}{2m}\sum_{i,j} \(A_ij - \frac{k_i k_j}{2m}\) * \detal_(c_i, c_j)
    Directed:
    .. math:: Q = \frac{1}{m}\sum_{i,j} \(A_ij - \frac{k_i^{in} k_j^{out}}{m}\) * \detal_{c_i, c_j}
    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        The network of interest
    community_dict : dict
        A dictionary to store the membership of each node
        Key is node and value is community index
    Returns
    -------
    float
        The modularity of `network` given `community_dict`
    '''

    Q = 0
    G = network.copy()
    nx.set_edge_attributes(G, {e:1 for e in G.edges}, 'weight')
    A = nx.to_scipy_sparse_matrix(G).astype(float)

    if type(G) == nx.Graph:
        # for undirected graphs, in and out treated as the same thing
        out_degree = in_degree = dict(nx.degree(G))
        M = 2.*(G.number_of_edges())
        print("Calculating modularity for undirected graph")
    elif type(G) == nx.DiGraph:
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())
        M = 1.*G.number_of_edges()
        print("Calculating modularity for directed graph")
    else:
        print('Invalid graph type')
        raise TypeError

    nodes = list(G)
    Q = np.sum([A[i,j] - in_degree[nodes[i]]*\
                         out_degree[nodes[j]]/M\
                 for i, j in product(range(len(nodes)),\
                                     range(len(nodes))) \
                if community_dict[nodes[i]] == community_dict[nodes[j]]])
    return Q / M

def get_mod_matrix(network, comm_nodes=None, B=None):
    '''
    This function computes the modularity matrix
    for a specific group in the network.
    (a.k.a., generalized modularity matrix)
    Specifically,
    .. math::
        B^g_{i,j} = B_ij - \delta_{ij} \sum_(k \in g) B_ik
        m = \abs[\Big]{E}
        B_ij = A_ij - \dfrac{k_i k_j}{2m}
        OR...
        B_ij = \(A_ij - \frac{k_i^{in} k_j^{out}}{m}
    When `comm_nodes` is None or all nodes in `network`, this reduces to :math:`B`
    Parameters
    ----------
    network : nx.Graph or nx.DiGraph
        The network of interest
    comm_nodes : iterable (list, np.array, or tuple)
        List of nodes that defines a community
    B : np.matrix
        Modularity matrix of `network`
    Returns
    -------
    np.matrix
        The modularity of `comm_nodes` within `network`
    '''

    if comm_nodes is None:
        comm_nodes = list(network)
        return get_base_modularity_matrix(network)

    if B is None:
        B = get_base_modularity_matrix(network)

    indices = comm_nodes
    B_g = B[indices, :][:, indices]
    rowcol = 0.5*(B_g.sum(axis=0) + B_g.sum(axis=1))
    B_hat_g = B_g - np.diag(rowcol)
#    # subset of mod matrix in g
#    indices = [list(network).index(u) for u in comm_nodes]
#    B_g = B[indices, :][:, indices]
#    #print 'Type of `B_g`:', type(B_g)
#
#    # B^g_(i,j) = B_ij - δ_ij * ∑_(k∈g) B_ik
#    # i, j ∈ g
#    B_hat_g = np.zeros((len(comm_nodes), len(comm_nodes)), dtype=float)
#
#    # ∑_(k∈g) B_ik
#    B_g_rowsum = np.asarray(B_g.sum(axis=1))[:, 0]
#    if type(network) == nx.Graph:
#        B_g_colsum = np.copy(B_g_rowsum)
#    elif type(network) == nx.DiGraph:
#        B_g_colsum = np.asarray(B_g.sum(axis=0))[0, :]
#
#    for i in range(B_hat_g.shape[0]):
#        for j in range(B_hat_g.shape[0]):
#            if i == j:
#                B_hat_g[i,j] = B_g[i,j] - 0.5 * (B_g_rowsum[i] + B_g_colsum[i])
#            else:
#                B_hat_g[i,j] = B_g[i,j]
#
#    if type(network) == nx.DiGraph:
#        B_hat_g = B_hat_g + B_hat_g.T

    return sparse.csc_matrix(B_hat_g)

def largest_eig(A):
    '''
        A wrapper over `scipy.linalg.eig` to produce
        largest eigval and eigvector for A when A.shape is small
    '''
    vals, vectors = eig(A.todense())
    real_indices = [idx for idx, val in enumerate(vals) if not bool(val.imag)]
    vals = [vals[i].real for i in range(len(real_indices))]
    vectors = [vectors[i] for i in range(len(real_indices))]
    max_idx = np.argsort(vals)[-1]
    return np.asarray([vals[max_idx]]), np.asarray([vectors[max_idx]]).T
