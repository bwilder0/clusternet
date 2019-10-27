import torch

def loss_modularity(mu, r, embeds, dist, bin_adj, mod, args):
    bin_adj_nodiag = bin_adj*(torch.ones(bin_adj.shape[0], bin_adj.shape[0]) - torch.eye(bin_adj.shape[0]))
    return (1./bin_adj_nodiag.sum()) * (r.t() @ mod @ r).trace()

def loss_kcenter(mu, r, embeds, dist, bin_adj, obj, args):
    if obj == None:
        return torch.tensor(0).float()
    x = torch.softmax(dist*args.kcentertemp, 0).sum(dim=1)
    x = 2*(torch.sigmoid(4*x) - 0.5)
    if x.sum() > args.K:
        x = args.K*x/x.sum()
    loss = obj(x)
    return loss
