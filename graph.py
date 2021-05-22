import networkx as nx
from random import shuffle
from random import random
import numpy as np

def sequence_on_graph(G):
    nodes = list(G.nodes())    
    # for (n,nbrdict) in G.adjacency():
        # for adjacent_n in nbrdict.keys():
        # nodes.append(n)
    shuffle(nodes)
    ordering = set()
    edge_ordering = []
    for node in nodes:
        ordering.add(node)
        sub_ordering = []
        edge_list = G[node]
        for adjacent_node in edge_list:
            if adjacent_node in ordering:
                sub_ordering.append((node,adjacent_node))
        shuffle(sub_ordering)
        edge_ordering.append(sub_ordering)
    return (nodes,edge_ordering)

def sequence_on_graph_geometric(G):
    nodes = list(G.nodes())    
    # for (n,nbrdict) in G.adjacency():
        # for adjacent_n in nbrdict.keys():
        # nodes.append(n)
    # shuffle(nodes)
    ordering = set()
    edge_ordering = []

    nodes_selected = []
    nodes_left = nodes

    probabilities = []
    p = 1/2
    for i in range(len(nodes)-1):
        probabilities.append(p)
        p*=1/2

    if(len(probabilities)!=0):
        probabilities.append(p*2)
    else:
        probabilities.append(1)

    while(len(nodes_left)!=0):
        selected_node = np.random.choice(nodes_left,p=probabilities)
        nodes_selected.append(selected_node)
        nodes_left.remove(selected_node)
        probabilities.pop()
        if(len(probabilities)!=0):
            probabilities[-1] *=2

    for node in nodes_selected:
        ordering.add(node)
        sub_ordering = []
        edge_list = G[node]
        for adjacent_node in edge_list:
            if adjacent_node in ordering:
                sub_ordering.append((node,adjacent_node))
        shuffle(sub_ordering)
        edge_ordering.append(sub_ordering)
    return (nodes_selected,edge_ordering)

def construct_graph(node_ordering,edge_ordering):
    G = nx.Graph()
    new_edge_ordering = []
    for edge_list in edge_ordering:
        for edge in edge_list:
            new_edge_ordering.append(tuple(edge))
    G.add_nodes_from(node_ordering)
    G.add_edges_from(new_edge_ordering)
    return G