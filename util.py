import torch

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    '''
    vec: [1, n_labels]
    '''
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def log_sum_exp_batch(vec, dim=0):
    '''
    vec: [n, n_labels, batch_size]
    return: [n, batch_size]
    
    vec: [n_labels, batch_size, n]
    return: [batch_size, n]
    '''
    n, n_labels, bs = vec.shape
    # [batch_size, n]
    max_score, _ = torch.max(vec, dim=dim) 
    # [n, n_labels, batch_size]
    max_score_broadcast = max_score.unsqueeze(dim).expand(vec.shape)
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=dim))
