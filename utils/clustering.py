import numpy as np
from scipy.optimize import linear_sum_assignment

def cosine(x, y, eps=1e-15):
    x = x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=eps)
    y = y / np.linalg.norm(y, axis=-1, keepdims=True).clip(min=eps)
    return np.dot(x, y.T)


def AHC(sim_mx, threshold=0):
    """ Performs UPGMA variant (wikipedia.org/wiki/UPGMA) of Agglomerative
    Hierarchical Clustering using the input pairwise similarity matrix.
    Input:
        sim_mx    - NxN pairwise similarity matrix
        threshold - threshold for stopping the clustering algorithm
                    (see function twoGMMcalib_lin for its estimation)
    Output:
        cluster labels stored in an array of length N containing (integers in
        the range from 0 to C-1, where C is the number of dicovered clusters)
    """
    dist = -sim_mx
    dist[np.diag_indices_from(dist)] = np.inf
    clsts = [[i] for i in range(len(dist))]
    while True:
        mi, mj = np.sort(np.unravel_index(dist.argmin(), dist.shape))
        if dist[mi, mj] > -threshold:
            break
        dist[:, mi] = dist[mi,:] = (dist[mi,:]*len(clsts[mi])+dist[mj,:]*len(clsts[mj]))/(len(clsts[mi])+len(clsts[mj]))
        dist[:, mj] = dist[mj,:] = np.inf
        clsts[mi].extend(clsts[mj])
        clsts[mj] = None
    labs= np.empty(len(dist), dtype=int)
    for i, c in enumerate([e for e in clsts if e]):
        labs[c] = i
    return labs


def AHC2(embds, threshold=0, score_fn=None):
    assert embds.ndim == 2
    seq_len = len(embds)
    if seq_len == 1:
        return np.array([0])
    if score_fn is None:
        score_fn = cosine

    sim_mx = score_fn(embds, embds)
    np.fill_diagonal(sim_mx, -np.inf)

    clsts = [[i] for i in range(seq_len)]
    clst_embds = embds.copy()
    while True:
        mi, mj = np.sort(np.unravel_index(sim_mx.argmax(), sim_mx.shape))
        if sim_mx[mi, mj] < threshold:
            break
        clsts[mi] += clsts[mj]
        clsts[mj] = None
        clst_embds[mi] = embds[clsts[mi]].mean(axis=0)
        ms = [i for i, e in enumerate(clsts) if e is not None]
        sim_mx[ms,mi] = sim_mx[mi,ms] = score_fn(clst_embds[mi], clst_embds[ms])
        sim_mx[:, mj] = sim_mx[mj, :] = sim_mx[mi, mi] = -np.inf
    labs = np.empty(seq_len, dtype=int)
    for i, c in enumerate([e for e in clsts if e]):
        labs[c] = i
    return labs

def AHC3(embds, lens, threshold=0, score_fn=None):
    assert embds.ndim == 2
    seq_len = len(embds)
    if seq_len == 1:
        return np.array([0])
    if score_fn is None:
        score_fn = cosine

    sim_mx = score_fn(embds, embds)
    np.fill_diagonal(sim_mx, -np.inf)

    clsts = [[i] for i in range(seq_len)]
    clst_embds = embds.copy()
    while True:
        mi, mj = np.sort(np.unravel_index(sim_mx.argmax(), sim_mx.shape))
        if sim_mx[mi, mj] < threshold:
            break
        clsts[mi] += clsts[mj]
        clsts[mj] = None
        # clst_embds[mi] = embds[clsts[mi]].mean(axis=0) # do not use mean, use weighted average
        clst_embds[mi] = (embds[clsts[mi]]*lens[clsts[mi]][:, None]).sum(axis=0) / lens[clsts[mi]].sum() # weighted average
        ms = [i for i, e in enumerate(clsts) if e is not None]
        sim_mx[ms,mi] = sim_mx[mi,ms] = score_fn(clst_embds[mi], clst_embds[ms])
        sim_mx[:, mj] = sim_mx[mj, :] = sim_mx[mi, mi] = -np.inf
    labs = np.empty(seq_len, dtype=int)
    for i, c in enumerate([e for e in clsts if e]):
        labs[c] = i
    return labs

def AAHC(embds, threshold=0, score_fn=None):
    ''' Adjacent Agglomerative Hierarchical Clustering
    '''
    assert embds.ndim == 2
    seq_len = len(embds)
    if seq_len == 1:
        return np.array([0])
    if score_fn is None:
        score_fn = cosine

    embds = [x.copy() for x in embds]
    scores = [score_fn(embds[i], embds[i+1]) for i in range(seq_len-1)]
    clsts = [[i] for i in range(seq_len)]
    while len(scores) > 0:
        i = max(enumerate(scores), key=lambda x: x[1])[0]   # argmax for list.
        if scores[i] < threshold:
            break

        embds[i] = embds[i+1] = (embds[i] * len(clsts[i]) + \
            embds[i+1] * len(clsts[i+1])) / (len(clsts[i]) + len(clsts[i+1]))
        if i + 1 < len(scores):
            scores[i+1] = score_fn(embds[i+1], embds[i+2])
        if i > 0:
            scores[i-1] = score_fn(embds[i-1], embds[i])
        clsts[i] = clsts[i+1] = clsts[i] + clsts[i+1]
        scores.pop(i)
        embds.pop(i)
        clsts.pop(i)

    labs = np.empty(seq_len, dtype=int)
    for i, c in enumerate(clsts):
        labs[c] = i
    return labs


def hungarian(labs1, labs2) -> list:

    dic = {}; cd = 0                                              # dic is the mapping betweeen labs2 and normalized labs2
    for i in labs2:
        if i not in dic.keys():
            dic[i] = cd; cd += 1
    labs_cache_encode = [dic[i] for i in labs2]                   # encoded labs_cache

    curr_cats  = len(set(labs1))                                  # number of cats in labs1
    if curr_cats == 0:
        labs1 = labs_cache_encode

    else:
        curr_cats_ = len(set(labs2[:-1]))                         # number of cats in labs2
        last_lab = labs_cache_encode[-1]
        
        if last_lab not in labs_cache_encode[:-1]:                # judge the last embedding belongs to a new cluster,
            labs1.append(max(labs1)+1)                            # if so, just add new lab to *labs1*

        # else, Hungarian algorithm
        else:
            cost_mtx = np.zeros((curr_cats_, curr_cats))          # construct the cost matrix
            for i, j in zip(labs_cache_encode[:-1], labs1):
                cost_mtx[i][j] += 1
            row_ind, col_ind = linear_sum_assignment(cost_mtx, maximize=True)
            if last_lab not in row_ind:
                labs1.append(max(labs1)+1)
            else:
                labs1.append(col_ind[np.where(row_ind == last_lab)[0][0]])
    
    return labs1


def hungarian2(labs1, labs2, embds, seg_lens) -> list:

    dic = {}; cd = 0                                              # dic is the mapping betweeen labs2 and normalized labs2
    for i in labs2:
        if i not in dic.keys():
            dic[i] = cd; cd += 1
    labs_cache_encode = [dic[i] for i in labs2]                   # encoded labs_cache

    curr_cats  = len(set(labs1))                                  # number of cats in labs1
    if curr_cats == 0:
        labs1 = labs_cache_encode

    else:
        curr_cats_ = len(set(labs2[:-1]))                         # number of cats in labs2
        last_lab = labs_cache_encode[-1]
        
        if last_lab not in labs_cache_encode[:-1]:                # judge the last embedding belongs to a new cluster,
            labs1.append(max(labs1)+1)                            # if so, just add new lab to *labs1*

        # else, Hungarian algorithm
        else:
            cost_mtx = np.zeros((curr_cats_, curr_cats))          # construct the cost matrix
            np_labs1 = np.array(labs1); np_labs2 = np.array(labs_cache_encode[:-1])
            for i in range(curr_cats_):
                for j in range(curr_cats):
                    e1 = embds[:-1][np.where(np_labs2==i)].sum(axis=0) / seg_lens[:-1][np.where(np_labs2==i)].sum(axis=0)
                    e2 = embds[:-1][np.where(np_labs1==j)].sum(axis=0) / seg_lens[:-1][np.where(np_labs1==j)].sum(axis=0)
                    cost_mtx[i][j] = cosine(e1, e2)
            row_ind, col_ind = linear_sum_assignment(cost_mtx, maximize=True)
            if last_lab not in row_ind:
                labs1.append(max(labs1)+1)
            else:
                labs1.append(col_ind[np.where(row_ind == last_lab)[0][0]])
    
    return labs1


if __name__ == '__main__':
    np.set_printoptions(2, linewidth=np.inf)

    from sklearn.datasets import make_blobs
    centers = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    cluster_std = [0.5, 0.5, 0.5, 0.5]
    X, Y = make_blobs(n_samples=20, cluster_std=cluster_std, centers=centers, n_features=2, random_state=1)
    norm_X = X / np.linalg.norm(X, axis=1, keepdims=True)
    sim_mx = np.dot(norm_X, norm_X.T)
    print(Y)

    # labs = AHC(sim_mx, threshold=0.5)
    labs = AHC2(X, threshold=0.5)
    print(labs)
    labs = AAHC(X, threshold=0.5)
    print(labs)
