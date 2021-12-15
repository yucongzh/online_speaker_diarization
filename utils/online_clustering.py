from re import T
import numpy as np
from numpy.core.fromnumeric import nonzero
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment

def cosine(x, y, eps=1e-15):
    # x, y = np.array(x), np.array(y)
    x = x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=eps)
    y = y / np.linalg.norm(y, axis=-1, keepdims=True).clip(min=eps)
    return np.dot(x, y.T)

def metric1(embd, clst_embds, score_fn, method="centroid"):
    '''
    return 1*C np.array, each elements reflects the similarity between \
        the embedding and one cluster
    '''
    if method == "centroid":
        clsts = [np.mean(c, axis=0) for c in clst_embds]
        return score_fn(embd, clsts)
    elif method == "adaptivePCA":
        clsts = [np.mean(score_fn(c, embd)) for c in clst_embds]
        return np.array(clsts)

def metric(embd, clst_embds, score_fn):
    '''
    return 1*C np.array, each elements reflects the similarity between \
        the embedding and one cluster
    '''
    scores = [score_fn(embd, c) for c in clst_embds]
    return np.array(scores)


def AHC(embds, threshold=0, score_fn=None, method="centroid", R=8, n_component=1):
    '''
    method: default "centroid"
        "centroid": each time use similarity of average embed as metric \
            to measure two clusters
        "single": each time use max similarity as metric to measure two clusters
        "adaptivePCA": use transformation matrix to adaptively emphasis the subspace
    '''

    assert embds.ndim == 2
    
    if score_fn is None:
        score_fn = cosine

    clsts       = [0]                     # adaptive clusters, starting from on cluster, dim: 1*C
    clsts_embds = [[embds[0]]]            # save all embeddings along the clusters, dim: C*N*D
    clsts       = [i for i in range(len(clsts_embds))]
    labs        = [0]                       # results of resegmentation
    embd_sz     = embds.shape[1]

    # simulate the online process
    for embd in embds[1:]:

        clsts_embds_copy = clsts_embds.copy()

        if method == "adaptivePCA":
            pca = PCA(n_components=n_component)

            for i in range(len(clsts_embds_copy)):
                
                clst = np.array(clsts_embds_copy[i])

                # compute cluster size
                clst_sz, embd_sz = clst.shape

                # compute first principle component
                V = pca.fit_transform(clst.T)

                # compute alpha and transformation matrix
                alpha = clst_sz / (clst_sz + R)
                T = alpha * V @ V.T + (1 - alpha) * np.identity(embd_sz)

                # apply the transform matrix to all the current embds
                clsts_embds_copy[i] = (clst @ T).tolist()

        # compute the similarity with all the current known clusters
        sim_list = metric1(embd, clsts_embds_copy, score_fn, method)
        mi = np.array(sim_list).argmax()

        # compare with the threshold. If max(s) < t, add new cluster
        if sim_list[mi] < threshold:
            clsts.append(len(clsts))
            clsts_embds.append([embd])
            # np.append(clsts_embds, [[embd]], axis=0)
            labs.append(clsts[-1])

        # O.W. merge to the cluster with the largest s
        else:
            clsts_embds[mi].append(embd)
            # np.append(clsts_embds[mi], [embd], axis=0)
            labs.append(mi)

    return np.array(labs)

def AAHC(embds, threshold=0, score_fn=None):
    assert embds.ndim == 2
    if score_fn is None:
        score_fn = cosine

    labs        = [0]                     # results of resegmentation
    last        = embds[0]                # record the last embedding
    cluster_id  = 0                       # cluster id
    embd_sum    = embds[0]
    cnt         = 1

    # simulate the online process
    for embd in embds[1:]:
        if score_fn(embd, last) < threshold:
            cluster_id += 1
            embd_sum, cnt = embd, 1
        else:
            embd_sum += embd
            cnt += 1
        labs.append(cluster_id)
        last = embd_sum / cnt

    return np.array(labs)

# original offline clustering
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
        curr_cats_ = len(set(labs2[:-1]))                # number of cats in labs2
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

def hungarian1(labs1, labs2, embds, seg_lens) -> list:

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
    # print(X)
    print(Y)

    labs = AHC(X, threshold=0.5, method="centroid")
    print(labs)

    labs = AHC(X, threshold=0.5, method="adaptivePCA")
    print(labs)

    # labs = AHC2(X, threshold=0.5)
    # print(labs)
