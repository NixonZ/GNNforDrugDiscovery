from graph import *
from data import *
from model import *
import networkx as nx
import torch.optim as optim

# Molecules for testing
mols = []
mol = 'C12=C3C4=C5C6=C1C7=C8C9=C1C%10=C%11C(=C29)C3=C2C3=C4C4=C5C5=C9C6=C7C6=C7C8=C1C1=C8C%10=C%10C%11=C2C2=C3C3=C4C4=C5C5=C%11C%12=C(C6=C95)C7=C1C1=C%12C5=C%11C4=C3C3=C5C(=C81)C%10=C23'
mols.append(mol)
mol = "OC[C@@H](O1)[C@@H](O)[C@H](O)[C@@H]2[C@@H]1c3c(O)c(OC)c(O)cc3C(=O)O2"
mols.append(mol)
mol = "C1=CC=[C-]C=C1.[Mg+2].[Br-]"
mols.append(mol)
mol = "C1C(=O)NC2=C(C=C(C=C2)Br)C(=N1)C3=CC=CC=C3Cl"
mols.append(mol)
mol = "C1=CC=C2C(=C1)C(=O)O[Bi]O2.O"
mols.append(mol)
mol = "CNC[C@@H]([C@H]([C@@H]([C@@H](CO)O)O)O)O.CNC[C@@H]([C@H]([C@@H]([C@@H](CO)O)O)O)O.C1=CC=C(C=C1)COCC(C(=O)[O-])N(CCN(CCN(CC(=O)O)CC(=O)[O-])CC(=O)[O-])CC(=O)O.[Gd+3]"
mols.append(mol)
mol = "COC(CNC(=O)N)C[Hg]Cl"
mols.append(mol)
mols = [ read_molecule_mol(mol) for mol in mols ]

# Create a torch-geometric dataloader
dataset = []
nx_dataset = []
for mol in mols:
    G = mol_to_nx(mol)
    nx_dataset.append(G)
    data = nx_to_torch_geom(G).to(device)
    dataset.append(data)
loader = DataLoader(dataset, batch_size= 1)

# Sequence to tensor format
def sequence_tensor(sequence):
    node_ordering = torch.tensor(sequence[0]).to(device)

    edge_ordering = sequence[1]
    max_len = 0
    for edges in edge_ordering:
        if len(edges) > max_len:
            max_len = len(edges)
    
    for i,_ in enumerate(edge_ordering):
        while( len(edge_ordering[i])!= max_len ):
            edge_ordering[i].append((-1,-1,-1))
    edge_ordering = torch.tensor(edge_ordering).to(device)
    return (node_ordering,edge_ordering)

print(device)

# Create Model
model = Model(128,47,250,5)
model.to(device)

model.train()
# Parameters
epochs = 300
losses = []
torch.autograd.set_detect_anomaly(True)
lr = 0.000002
optimizer = optim.SGD(model.parameters(recurse=True), lr=lr, momentum=0.09)
verbose = False
for i in range(epochs):
    loss = torch.zeros((1),dtype=torch.float).to(device)
    optimizer.zero_grad()
    j = 0
    for batch in loader.dataset:
        sequence = sequence_on_graph(nx_dataset[j])
        n = torch.tensor( list(range(2, len(sequence[0])+1) ) ).to(device)
        n = torch.sum(torch.log(n))
        loss += model.forward(batch,sequence,verbose) + n
        j+=1
        if i%10==0:
            print(i,loss.item())
    if i%100==0:
        verbose = True
    else:
        verbose = False
    losses.append(loss.item())
    loss.backward(retain_graph=True)
    loss.detach_()
    loss = loss.detach()
    optimizer.step()
    optimizer.zero_grad()