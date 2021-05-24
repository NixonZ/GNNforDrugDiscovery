import networkx as nx
from random import shuffle
from random import random
import numpy as np
# from pubchempy import get_compounds, Compound
from RDkit.rdkit import Chem

# def read_molecule_from_pubchem_id(pubchem_id:str) -> nx.Graph:
#     mol = Compound.from_cid(pubchem_id)
#     return read_molecule(mol.canonical_smiles)
    

def read_molecule(smiles_rep: str):
    # G = read_smiles(smiles_rep)
    G = Chem.MolFromSmiles(smiles_rep)
    return G

def sequence_on_graph(G: nx.Graph):
    '''
    Generates a random ordering of nodes and edges in Uniform distribution
    input: A grapg G of type nx.Graph()
    Output: List of nodes in order of node ordering and a list of list of tuples of edge orders.
    '''
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

def sequence_on_graph_geometric(G: nx.Graph):

    '''
    Generates a random ordering of nodes and edges in Geometric distribution (Restricted Sample space)
    input: A graph G of type nx.Graph()
    Output: List of nodes in order of node ordering and a list of list of tuples of edge orders.
    '''
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

def construct_graph(node_ordering,edge_ordering) -> nx.Graph:
    '''
    Constructs the graph, given its node ordering and edge ordering
    inputs:
    @params node_ordering: List of nodes in order of node_ordering
    @params edge_ordering: List of list of tuples of edges in edge ordering following the node_ordering
    Outputs:
    A graph G of type nx.Graph()
    '''
    G = nx.Graph()
    new_edge_ordering = []
    for edge_list in edge_ordering:
        for edge in edge_list:
            new_edge_ordering.append(tuple(edge))
    G.add_nodes_from(node_ordering)
    G.add_edges_from(new_edge_ordering)
    return G

def valid_molecule(G: nx.Graph) -> bool :
    '''
    Checks all valency constraints are satisfied
    Valency chosen for validity is the common valency shown by the atoms
    C(4), N(3), O(2), X(1) and so on
    '''
    valid = True
    for (node,node_attr_dict) in G.nodes(data=True):
        num_H = int(node_attr_dict['hcount'])
        count = num_H
        for edge, edge_attr_dict in G[node]:
                count += edge_attr_dict['order']
        count = math.ceil(count)
        if node_attr_dict['element'] == 'C':
            if count-node_attr_dict['charge'] != 4:
                return False
        elif node_attr_dict['element'] == 'O':
            if count-node_attr_dict['charge'] != 2:
                return False
    return valid