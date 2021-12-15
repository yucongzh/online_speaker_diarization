import numpy as np
import pandas as pd

import networkx as nx
from networkx.algorithms.approximation.clique import max_clique

## construct a graph using labels
def construct_graph(labs):
    G = nx.Graph(); num_clsts = len(labs)
    for i in range(num_clsts):
        clst_sz = len(labs[i])
        for p in range(clst_sz):
            G.add_node(labs[i][p])
            for q in range(p+1, clst_sz):
                G.add_node(labs[i][q])
                G.add_edge(labs[i][p], labs[i][q])
    return G

def get_embds_from_graph(G) -> list:
    mydic = dict(G.nodes.data()).values()
    df = pd.DataFrame.from_dict(mydic)
    return df['embd'].tolist()

def get_start_from_graph(G) -> list:
    mydic = dict(G.nodes.data()).values()
    df = pd.DataFrame.from_dict(mydic)
    return df['start'].tolist()

def get_end_from_graph(G) -> list:
    mydic = dict(G.nodes.data()).values()
    df = pd.DataFrame.from_dict(mydic)
    return df['end'].tolist()

def get_duration_from_graph(G) -> list:
    mydic = dict(G.nodes.data()).values()
    df = pd.DataFrame.from_dict(mydic)
    return [round(ed-st, 2) for st, ed in zip(df['start'].tolist(), df['end'].tolist())]

## given graph and embeddings, find the best embds that can represent the graph
def compute_centroid(G, embds, durations):
    core_labs = list(max_clique(G))
    return (embds[core_labs]*durations[core_labs, np.newaxis]).sum(axis=0) / durations[core_labs].sum()
