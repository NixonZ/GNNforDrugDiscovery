# GNNforDrugDiscovery
SURA 2021

## graph.py
Contains all the required operations to be performed on our graph.
The various methods in this file are:
1. __read_molecule_from_pubchem_id__: This returns networkx.Graph representation of the molecule given its pubchemId.
2. __mol_to_nx__: Takes input as a molecule in rdkit.Mol class and gives the networkx.Graph representation of the molecule.
3. __nx_to_mol__: Takes input as nx.Graph representation of a molecule and outputs rdkit.Mol representation of the molecule.
4. __read_molecule_nx__: Takes input as Canonical smiles representation of the molecule and outputs the nx.Graph representation of the molecule.
5. __read_molecule_mol__: Takes input as Canonical smiles representation of the molecule and outputs the rdkit.Mol representation of the molecule.
6. __sequence_on_graph__: Takes input as nx.Graph and gives a node ordering and edge ordering of the graph following uniform distribution over the permutation of vertices.
7. __sequence_on_graph_geometric__: Takes input as nx.Graph and gives a node ordering and edge ordering of the graph following geometric distribution over the permutation of the vertices. The details of the distribution are provided in a separate section of this file.
8. __construct_graph__: Takes input as node ordering and edge ordering , and outputs the nx.Graph representation which has the input ordering as one of its orderings.
9. __valid_molecule__: Checks the validity of input graph(as a molecule) in nx.Graph form.
10. __generate_graph_from_sequence__:Takes input as node ordering and edge ordering in the format as output by __sequence_on_graph__, and outputs the nx.Graph representation which has the input ordering as one of its orderings.

## data.py
Contains conversion methods from rdkit.mol to torch.geometric.data and nx.Graph to torch.geometric.data and vice versa
The various methods and global variables in the file are:
1. __hybridization_types__: A dictionary mapping hybridisation type to an index.
2. __chiral_types__: A dictionary mapping chiral type to an index.
3. __bond_types__: A dictionary mapping bond types to an index.
4. __bond_dirs__: A dictionary mapping bond directions to and index.
5. __bond_steroes__: A dictionary mapping bond stereo to and index.
6. __torch_geom_to_mol__: Takes input as a torch_geometric.data as input and output rdkit.mol representation of the molecule.
7. __nx_to_torch_geom__: Given an nx.Graph as input outputs torch_geometric.data representation of the graph.
8. __mol_to_torch_geom__: Given an rdkit.Mol as input outputs torch_geometric.data representation of the graph.
9. __read_graphs_from_datase__: Reads dataset which is stored locally in directory(Given as input).

## model.py
Contains the model implementation
The description of various methods and classes are as follows.
1. __MPN(MessagePassing)__: This is the message passing layer of our Graph Neural network.
2. __Graph_Representation(nn.Module)__: This is the propagation layer of our GNN.
3. __f_addnode(nn.Module)__: Neural network that decides probability of adding nodes to the subgraph.
4. __f_add_edge(nn.Module)__: Neural network that decides probability of adding edges to the subgraph connecting to the recently added node.
5. __f_nodes(nn.Module)__: Neural network that decides probability of adding edge to the subgraph connecting to the recently added node over the nodes of the rest of the subgraph.
6. __Model(nn.Module)__: The final GNN model. 
