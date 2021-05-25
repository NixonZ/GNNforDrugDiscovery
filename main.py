from graph import *
import networkx as nx

G = nx.path_graph(4)

print(sequence_on_graph(G))

mol = read_molecule_mol("OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H]2[C@@H]1c3c(O)c(OC)c(O)cc3C(=O)O2")
# print(read_molecule_from_pubchem_id('25004'))
G = mol_to_nx(mol)

# nodes in the graph
for k,v in G.nodes(data=True):
    print(k,v)

# edges in the graph
for u,k,v in G.edges(data=True):
    print(u,k,v)