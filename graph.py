import networkx as nx
from random import shuffle
from random import random
import numpy as np
import math
from pubchempy import get_compounds, Compound
from RDkit.rdkit import Chem
from RDkit.rdkit.Chem import rdchem

def read_molecule_from_pubchem_id(pubchem_id:str) -> nx.Graph:
    mol = Compound.from_cid(pubchem_id)
    return read_molecule_mol(mol.canonical_smiles)
    
def mol_to_nx(mol: rdchem.Mol) -> nx.Graph:
    '''
    @Source: https://github.com/dakoner/keras-molecules/blob/dbbb790e74e406faa70b13e8be8104d9e938eba2/convert_rdkit_to_networkx.py
    input: Chem.rdchem.Mol type object
    Output: Networx Graph of the molecule.
    '''
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                   element = atom.GetSymbol(),
                   atomic_num=atom.GetAtomicNum(),
                   formal_charge=atom.GetFormalCharge(),
                   chiral_tag=atom.GetChiralTag(),
                   hybridization=atom.GetHybridization(),
                   num_explicit_hs=atom.GetNumExplicitHs(),
                   is_aromatic=atom.GetIsAromatic())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                   bond.GetEndAtomIdx(),
                   bond_type=bond.GetBondType(),
                   is_aromatic=bond.GetIsAromatic(),
                   conjugation=bond.GetIsConjugated(),
                   stereochemistry=bond.GetStereo(),
                   bond_dir=bond.GetBondDir(),
                   is_ring=bond.IsInRing(),
                   is_in_ring_size_5=bond.IsInRingSize(5),
                   is_in_ring_size_6=bond.IsInRingSize(6),
                   is_in_ring_size_3=bond.IsInRingSize(3),
                   is_in_ring_size_7=bond.IsInRingSize(7))
    return G

def nx_to_mol(G: nx.Graph) -> rdchem.Mol:
    '''
    @source: https://github.com/dakoner/keras-molecules/blob/dbbb790e74e406faa70b13e8be8104d9e938eba2/convert_rdkit_to_networkx.py
    input: A networX graph representation of a molecule
    output: Chem.rdchem.Mol object of the corresponding molecule.
    '''
    mol = Chem.RWMol()
    atomic_nums = nx.get_node_attributes(G, 'atomic_num')
    chiral_tags = nx.get_node_attributes(G, 'chiral_tag')
    formal_charges = nx.get_node_attributes(G, 'formal_charge')
    node_is_aromatics = nx.get_node_attributes(G, 'is_aromatic')
    node_hybridizations = nx.get_node_attributes(G, 'hybridization')
    num_explicit_hss = nx.get_node_attributes(G, 'num_explicit_hs')
    node_to_idx = {}
    for node in G.nodes():
        a=Chem.Atom(atomic_nums[node])
        a.SetChiralTag(chiral_tags[node])
        a.SetFormalCharge(formal_charges[node])
        a.SetIsAromatic(node_is_aromatics[node])
        a.SetHybridization(node_hybridizations[node])
        a.SetNumExplicitHs(num_explicit_hss[node])
        idx = mol.AddAtom(a)
        node_to_idx[node] = idx

    bond_types = nx.get_edge_attributes(G, 'bond_type')
    for edge in G.edges():
        first, second = edge
        ifirst = node_to_idx[first]
        isecond = node_to_idx[second]
        bond_type = bond_types[first, second]
        mol.AddBond(ifirst, isecond, bond_type)

    Chem.SanitizeMol(mol)
    return mol

def read_molecule_nx(smiles_rep: str) -> nx.Graph:
    # G = read_smiles(smiles_rep)
    G = Chem.MolFromSmiles(smiles_rep)
    return mol_to_nx(G)

def read_molecule_mol(smiles_rep: str) -> rdchem.Mol:
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
# def Get_log_Probability(base_node_ordering, node_ordering):
    
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

def generate_sequence_from_graph(G: nx.Graph,node_ordering,edge_ordering) -> nx.Graph:
    G_ = nx.Graph()
    for i,node in enumerate(node_ordering):
        G_.add_nodes_from([(node,G.nodes[node])])
        for edge in edge_ordering[i]:
            G_.add_edges_from( [ (edge[0],edge[1],G.edges[edge]) ] )
    return G_

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

def construct_graph(node_ordering,edge_ordering):
    G = nx.Graph()
    new_edge_ordering = []
    for edge_list in edge_ordering:
        for edge in edge_list:
            new_edge_ordering.append(tuple(edge))
    G.add_nodes_from(node_ordering)
    G.add_edges_from(new_edge_ordering)
    return G

def generate_sequence_from_graph(G: nx.Graph,node_ordering,edge_ordering) -> nx.Graph:
    G_ = nx.Graph()
    for i,node in enumerate(node_ordering):
        G_.add_nodes_from([(node,G.nodes[node])])
        for edge in edge_ordering[i]:
            G_.add_edges_from( [ (edge[0],edge[1],G.edges[edge]) ] )
    return G_