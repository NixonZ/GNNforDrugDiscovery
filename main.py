from graph import *
import networkx as nx

G = nx.path_graph(4)

print(sequence_on_graph(G))

print(read_molecule("OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H]2[C@@H]1c3c(O)c(OC)c(O)cc3C(=O)O2"))
# print(read_molecule_from_pubchem_id('25004'))