from sklearn.decomposition import PCA
from utils.clustering import cosine
import numpy as np
import pandas as pd
import networkx as nx
import math, re
from tqdm import tqdm
from copy import deepcopy
from random import choice
from utils.graph_method import *
from networkx.classes.function import subgraph

import time

class lab_master(object):
    
    def __init__(self):
        self.seg_embds_origin = []
        self.seg_embds        = []
        self.seg_lens         = []
        self.seg_ts           = []     # keeps all the segments' time, e.g. [[s0,e0], [s1,e1], ...]
        self.centroids        = []     # keeps all the centroid value of each cluster, e.g. [c1, c2, c3, c4]
        self.clsts_labels     = []     # only keep embeddings' labels of each cluster, e.g. [[0,1,2], [3], [4,5], [6]]
        self.size             = 0

        self.labs             = []

        self.graph            = nx.Graph()
        self.graph_clsts      = []     # clusters using 0.3 as threshold


    def compute_labels(self):
        # print('computing labels...')
        labs = np.empty(self.size, dtype=int)
        for i, c in enumerate(self.clsts_labels):
            labs[c] = i
        return labs


    ## agglomerative hierachy clustering
    def ahc(self, clst_thres):
        clst_embds = np.array(self.centroids.copy()); clsts = self.clsts_labels.copy(); clsts_id = [[i] for i in range(len(clst_embds))]
        seg_lens = np.array([np.array(self.seg_lens)[c].sum() for c in self.clsts_labels])
        sim_mx = cosine(clst_embds, clst_embds); np.fill_diagonal(sim_mx, -np.inf)          # construct similarity matrix
        while True:
            mi, mj = np.sort(np.unravel_index(sim_mx.argmax(), sim_mx.shape))               # find the best pair
            if sim_mx[mi, mj] < clst_thres:                                                 # stop if max similarity smaller than a threshold
                break
            clsts[mi] = clsts[mi] + clsts[mj]; clsts[mj] = None                             # update ith and jth clsts
            clsts_id[mi] = clsts_id[mi] + clsts_id[mj]; clsts_id[mj] = None
            clst_embds[mi] = (clst_embds[clsts_id[mi]] * seg_lens[clsts_id[mi], np.newaxis]).sum(axis=0) \
                / seg_lens[clsts_id[mi]].sum()                                              # re-calculate ith clst embd
            seg_lens[mi] = seg_lens[mi] + seg_lens[mj]                                      # update segment_length
            ms = [i for i, e in enumerate(clsts_id) if e is not None]                       # update all the scores with new ith clst embd
            sim_mx[ms,mi] = sim_mx[mi,ms] = cosine(clst_embds[mi], clst_embds[ms])          # update...
            sim_mx[:, mj] = sim_mx[mj, :] = sim_mx[mi, mi] = -np.inf                        # update...
        labs = np.empty(self.size, dtype=int)
        for i, c in enumerate([e for e in clsts if e]):
            labs[c] = i
        self.labs = labs


    ## checkpoint-ahc, with at least *maxclsts*
    def chkpt_ahc(self, clst_thres, maxclsts):
        clst_embds = np.array(self.centroids.copy()); clsts = self.clsts_labels.copy(); clsts_id = [[i] for i in range(len(clst_embds))]
        seg_lens = np.array([np.array(self.seg_lens)[c].sum() for c in self.clsts_labels])
        sim_mx = cosine(clst_embds, clst_embds); np.fill_diagonal(sim_mx, -np.inf)          # construct similarity matrix
        while True:
            mi, mj = np.sort(np.unravel_index(sim_mx.argmax(), sim_mx.shape))               # find the best pair
            tmp = list([c for c in clsts if c]); tmp2 = list([c for c in clsts_id if c])
            ## update and store the current state
            if len(tmp) == maxclsts or (len(tmp) > maxclsts and sim_mx[mi, mj] < clst_thres): # if the similarity score is too small or reach to *maxclsts*
                self.clsts_labels = tmp; self.centroids = [clst_embds[c[0]] for c in tmp2]
            if sim_mx[mi, mj] < clst_thres:                                                 # stop if max similarity smaller than a threshold
                break
            clsts[mi] = clsts[mi] + clsts[mj]; clsts[mj] = None                             # update ith and jth clsts
            clsts_id[mi] = clsts_id[mi] + clsts_id[mj]; clsts_id[mj] = None                 # update ith and jth clsts_id
            clst_embds[mi] = (clst_embds[clsts_id[mi]] * seg_lens[clsts_id[mi], np.newaxis]).sum(axis=0) \
                / seg_lens[clsts_id[mi]].sum()                                              # re-calculate ith clst embd
            seg_lens[mi] = seg_lens[mi] + seg_lens[mj]                                      # update segment_length
            ms = [i for i, e in enumerate(clsts_id) if e is not None]                       # update all the scores with new ith clst embd
            sim_mx[ms,mi] = sim_mx[mi,ms] = cosine(clst_embds[mi], clst_embds[ms])          # update...
            sim_mx[:, mj] = sim_mx[mj, :] = sim_mx[mi, mi] = -np.inf                        # update...
        labs = np.empty(self.size, dtype=int)
        for i, c in enumerate([e for e in clsts if e]):
            labs[c] = i
        self.labs = labs


    ## aahc segmentation
    @staticmethod
    def aahc_seg(embds, threshold=0, score_fn=None):
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

            # embds[i] = embds[i+1] = (embds[i] * seg_lens[clsts[i], np.newaxis]).sum(axis=0) \
            #     + (embds[i+1] * seg_lens[clsts[i+1], np.newaxis]).sum(axis=0) \
            #     / seg_lens[clsts[i]].sum()
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
        return clsts


    ## graph-based reclustering method
    def reclustering(self, spklen):
        # print("stage 1")
        labs = self.labs
        n_clsts = labs.max() + 1
        spks = []; non_spks = []; clst_embds_new = []; clsts_id = []; maxlen = float('-inf'); maxID = 0
        clsts_id = [np.where(labs == i)[0].tolist() for i in range(n_clsts)]
        for i in range(n_clsts):
            lens = np.array(self.seg_lens)[np.where(labs == i)]
            clst_len = lens.sum()
            if clst_len > maxlen:
                maxlen = clst_len
                maxID = i
            embds_new = np.array(self.seg_embds)[np.where(labs == i)]
            clst_embd = (embds_new * lens.reshape(-1, 1)).sum(axis=0) / clst_len
            clst_embds_new.append(clst_embd)
            if clst_len > spklen:
                spks.append(i)
            else:
                non_spks.append(i)
        # print(spks, non_spks)
        clst_embds_new = np.array(clst_embds_new)

        if len(spks) == 0: spks.append(maxID)
        
        ## Next step is to cluster all the other small non-spk clusters into those clusters
        ## Assign non-spk clusters to the best matching spk clusters in a naive way
        for non_spk in non_spks:
            max_score = 0.3
            assg_spk = -1
            for spk in spks:
                score = cosine(clst_embds_new[spk], clst_embds_new[non_spk])
                if score > max_score:
                    max_score = score
                    assg_spk = spk

            if assg_spk != -1:
                labs[np.where(labs == non_spk)] = assg_spk
            else:
                ## using connected ratio
                # neighbors = []
                # for l in clsts_id[non_spk]: neighbors += list(self.graph.neighbors(l))
                # neighbors = set(neighbors) # remove duplicates
                # max_ratio = 0; ratio = 0
                # for spk in spks:
                #     spk_set = set(clsts_id[spk])
                #     if len(neighbors) != 0:
                #         ratio = len(neighbors & spk_set) / len(neighbors)
                #     if ratio > max_ratio:
                #         max_ratio = ratio
                #         assg_spk = spk
                
                ## using average linkage
                neighbors = []
                for l in clsts_id[non_spk]: neighbors += list(self.graph.neighbors(l))
                neighbors = set(neighbors) # remove duplicates
                max_ratio = 0; ratio = 0
                for spk in spks:
                    weight = 0; spk_set = set(clsts_id[spk]); overlap_set = neighbors & spk_set
                    cnt = 0
                    for i in clsts_id[non_spk]:
                        for j in overlap_set:
                            if self.graph.has_edge(i,j):
                                weight += self.graph.edges[i, j]['weight']
                                cnt += 1
                    if cnt != 0: ratio = weight / cnt
                    if ratio > max_ratio:
                        max_ratio = ratio
                        assg_spk = spk
                        
                labs[np.where(labs == non_spk)] = assg_spk

        self.labs = labs

                        
    def read(self, embd, st, ed, seg_thres, graph_thres, model=None):
        ''' update embeddings
        '''
        # print("reading...")
        self.seg_embds_origin.append(embd); self.seg_embds.append(embd)
        self.seg_lens.append(ed-st); self.seg_ts.append([st, ed]); self.size += 1

        ## construct a graph with prunning 
        self.graph.add_node(self.size-1); sim_mx = cosine(embd, np.array(self.seg_embds))
        idx = [i for i in range(self.size-1)]
        for i in idx:
            if sim_mx[i] < graph_thres:
                for j in self.graph.neighbors(i):
                    if j in idx: idx.remove(j)
                continue
            self.graph.add_edge(i, self.size-1, weight=sim_mx[i])

        new_embds = np.array(self.seg_embds_origin.copy())

        ''' aahc scd, consider the newest embd
            update centroids, clsts and clsts_labels
            original aahc, only consider the incoming  embd and the last cluster
        '''
        clst_id = self.labs[-1] if self.labs != [] else None
        sim_score = cosine(new_embds[-1], self.centroids[clst_id]) if clst_id != None else float('-inf')
        if self.centroids != [] and sim_score >= seg_thres:                 # already exists clusters and similar
            self.clsts_labels[clst_id].append(self.size-1)
            c = self.clsts_labels[clst_id]                                  # get the embd_ids of the last clst, including the newest embd
            ## check if use dnn models to extract embeddings
            if model != None:
                input = []
                for embd_id in c: # for each embd_id in the cluster
                    start, end = self.seg_ts[embd_id]
                    s = int(start * sr); e = int(end * sr)
                    input.append(y[s:e])
                input = np.concatenate(input)
                if len(input) < 8000:
                    N     = math.ceil(8000 / len(input))
                    input = np.concatenate([input] * N)[:8000]
                self.centroids[clst_id]  = model(input, sr)                 # obtain the new embedding using dnn
            else:
                self.centroids[clst_id] = (new_embds[c]*np.array(self.seg_lens)[c, np.newaxis]).sum(axis=0) / np.array(self.seg_lens)[c].sum()
        else:                                                               # fresh start, no clusters / potentially new speaker detected
            self.clsts_labels.append([self.size-1]); self.centroids.append(new_embds[-1])


    def update(self, clst_thres, spklen, maxclsts=None, method="ahc"):
        ''' perform clustering and output labels
        '''
        ahc_st = time.time()
        # print("updating...")
        if method == "naive":
            self.nc(clst_thres)
        elif method == "ahc":
            self.ahc(clst_thres)
        elif method == "chkpt-ahc":
            self.chkpt_ahc(clst_thres, maxclsts) 
        ahc_ed = time.time()
            
        ## Technically speaking, the clusters are very pure here.
        ## Allow large number of small clusters
        ## do the reclustering to shrink the clusters
        # print("reclustering...")
        reclst_st = time.time()
        self.reclustering(spklen)
        reclst_ed = time.time()
        
        return ahc_ed-ahc_st, reclst_ed-reclst_st
