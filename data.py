from graph import *
import pandas as pd 
from torch_geometric.data import Data, DataLoader
import torch

hybridization_types = {'S':0,'SP':1,'SP2':2,'SP3':3,'SP3D':4,'SP3D2':5,'OTHER':6}
chiral_types = {'CHI_UNSPECIFIED':0,'CHI_TETRAHEDRAL_CW':1,'CHI_TETRAHEDRAL_CCW':2,'CHI_TETRAHEDRAL_OTHER':3}

bond_types_list = [ str(bond_type[1]) for bond_type in rdchem.BondType.values.items() if str(bond_type[1])!='UNSPECIFIED' and str(bond_type[1])!='ZERO' ]
bond_types = dict( (j,i) for i,j in enumerate(bond_types_list))

bond_dir_list = [ str(bond_dir[1]) for bond_dir in rdchem.BondDir.values.items()]
bond_dirs = dict( (j,i) for i,j in enumerate(bond_dir_list))

bond_stereo_list = [ str(bond_stereo[1]) for bond_stereo in rdchem.BondStereo.values.items() ]
bond_steroes = dict( (j,i) for i,j in enumerate(bond_stereo_list))

def torch_geom_to_mol(data: Data) -> rdchem.Mol:
    # To-do
    return None

def nx_to_torch_geom(G:nx.Graph) -> Data:
    
    # G = nx.convert_node_labels_to_integers(G)
    # G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.LongTensor(list(G.edges)).t().contiguous()

    data = {}

    data['edge_index'] = edge_index.view(2, -1)

    # constructing node embeddings
    x = []
    for k,v in G.nodes(data=True):
        node_embedding = [] # 128

        one_hot_element = [0]*114
        one_hot_element[v['atomic_num']-1] = 1
        node_embedding += one_hot_element

        one_hot_hybridization = [0]*7
        hybridization = v['hybridization']
        one_hot_hybridization[  hybridization_types[str(hybridization)] ] = 1
        node_embedding += one_hot_hybridization

        one_hot_chiral = [0]*4
        chiral_tag = v['chiral_tag']
        one_hot_chiral[  chiral_types[str(chiral_tag)] ] = 1
        node_embedding += one_hot_chiral

        node_embedding += [ v['formal_charge'] ]

        if v['is_aromatic']:
            node_embedding += [1,0]
        else:
            node_embedding += [0,1]

        x.append(node_embedding)

    data['x'] = torch.tensor(x,dtype=torch.float)

    # constructing edge embedding
    edge_attr = []
    for u,k,v in G.edges(data=True):
        edge_embedding = []
        
        one_hot_bond_type = [0]*20
        # print(bond_types_list)
        one_hot_bond_type[bond_types[str(v['bond_type'])]] = 1
        edge_embedding += one_hot_bond_type 

        one_hot_bond_dir = [0]*6
        one_hot_bond_dir[bond_dirs[str(v['bond_dir'])]] = 1
        edge_embedding += one_hot_bond_dir

        one_hot_bond_stereo = [0]*7
        one_hot_bond_stereo[bond_steroes[str(v['stereochemistry'])]] = 1
        edge_embedding += one_hot_bond_stereo

        if v['is_aromatic']:
            edge_embedding += [1,0]
        else:
            edge_embedding += [0,1]
        
        if v['conjugation']:
            edge_embedding += [1,0]
        else:
            edge_embedding += [0,1]
        
        if v['is_ring']:
            edge_embedding += [1,0]
        else:
            edge_embedding += [0,1]
        
        if v['is_in_ring_size_3']:
            edge_embedding += [1,0]
        else:
            edge_embedding += [0,1]
        
        if v['is_in_ring_size_5']:
            edge_embedding += [1,0]
        else:
            edge_embedding += [0,1]
        
        if v['is_in_ring_size_6']:
            edge_embedding += [1,0]
        else:
            edge_embedding += [0,1]

        if v['is_in_ring_size_7']:
            edge_embedding += [1,0]
        else:
            edge_embedding += [0,1]

        edge_attr.append(edge_embedding)

    data['edge_attr'] = torch.tensor(edge_attr,dtype=torch.float)

    data = Data.from_dict(data)

    return data

def mol_to_torch_geom(mol:rdchem.Mol) -> Data:
    return nx_to_torch_geom(mol_to_nx(mol))

def read_graphs_from_dataset(dir:str) -> DataLoader:
    '''
    Reads dataset which is stored locally in directory @param: dir
    '''
    data = pd.read_csv(dir)
    dataset = []
    for smiles in data["SMILES"]:
        G = read_molecule_nx(smiles)
        dataset.append(nx_to_torch_geom(G))
    loader = DataLoader(data)
    return loader