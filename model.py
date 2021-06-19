import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

from data import *
from mendeleev import element as ele
import matplotlib.pyplot as plt


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_default_device()

class MPN(MessagePassing):

    def __init__(self,node_embedding_dim,edge_embedding_dim):
        super(MPN,self).__init__(aggr="add")
        self.fe = nn.Sequential(
            nn.Linear(node_embedding_dim*2+edge_embedding_dim,node_embedding_dim),
            nn.ReLU(),
            nn.Linear(node_embedding_dim,node_embedding_dim)
        )

    def forward(self,x,edge_attr,edge_index):
        '''
        x : [|V|, node_embedding_dim]
        edge_attr : [|E|, edge_embedding_dim]
        edge_index : [2,|E|]
        '''
        return self.propagate(edge_index, x = x, edge_attr = edge_attr)

    def message(self,x_i,x_j,edge_attr):
        return self.fe(torch.cat([x_i,x_j,edge_attr],dim=1))

class Graph_Representation(nn.Module):

    def __init__(self,node_embedding_dim,edge_embedding_dim,graph_dim = 50,prop_steps = 2):
        super(Graph_Representation,self).__init__()

        # Message Passing Layers
        self.prop_steps = prop_steps
        # self.MPN = MPN(node_embedding_dim,edge_embedding_dim)
        self.MPN_list = nn.ModuleList( [ MPN(node_embedding_dim,edge_embedding_dim) for _ in range(prop_steps) ] )
        self.LSTM_list = nn.ModuleList( [ nn.LSTMCell(node_embedding_dim,node_embedding_dim) for _ in range(prop_steps) ] )

        # Learning Graph representation Layers
        self.gm = nn.Linear(node_embedding_dim,graph_dim)
        self.fm = nn.Linear(node_embedding_dim,graph_dim)

    def forward(self,batch):
        '''
        batch : Batch
        '''
        c = torch.zeros(batch.x.size()).to(device)
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        for i in range(self.prop_steps):
            x = batch.x
            a = self.MPN_list[i].forward(x,edge_attr,edge_index)
            batch.x,c = self.LSTM_list[i](x, (a,c) )

        x = batch.x
        g =  torch.sigmoid(self.gm(x))
        h_v_G = self.fm(x)
        h_G = torch.sum( g * h_v_G , dim = 0 )

        return h_G

class f_addnode(nn.Module):

    def __init__(self,graph_dim):
        super(f_addnode,self).__init__()
        
        self.fan = nn.Linear(graph_dim,115)

    def forward(self,h_G):

        return torch.softmax(self.fan(h_G),dim=0)

class f_add_edge(nn.Module):

    def __init__(self,node_embedding_dim,graph_dim):
        super(f_add_edge,self).__init__()
        
        self.fae = nn.Linear(graph_dim+node_embedding_dim,2)

    def forward(self,h_G,h_v):
        x = torch.cat([h_G,h_v],dim=0)
        return torch.softmax(self.fae(x),dim=0)

class f_nodes(nn.Module):

    def __init__(self,node_embedding_dim):
        super(f_nodes,self).__init__()
        
        self.fs = nn.Linear(2*node_embedding_dim,20)
    
    def forward(self,x,h_v):
        h_v = torch.broadcast_to(h_v,(x.size()[0],h_v.size()[0]))
        x = torch.cat([x,h_v],dim=1)
        s = self.fs(x)
        return torch.softmax(s,dim=1)

class new_node(nn.Module):

    def __init__(self,node_embedding_dim,edge_embedding_dim,graph_dim,prop_steps):
        super(new_node,self).__init__()

        self.R_init = Graph_Representation(node_embedding_dim,edge_embedding_dim,graph_dim,prop_steps)
        self.f_init = nn.Linear(graph_dim+114,node_embedding_dim)

    def forward(self,batch,x_v):
        x = torch.cat([self.R_init(batch),x_v],dim=0)
        x = self.f_init(x)
        # atm_num,hybd,chr_tag,is_arm = x[:115],x[115:123],x[123:127],x[127:128]
        # idx = torch.argmax(torch.softmax(atm_num))
        # x[:115] = torch.zeros(x.size())
        # x[:115][idx] = 1
        return x

'''
G = (V,E)
Step -by step Generation
We have to learn the probability that our model predicts this graph and its ordering
'''

class Model(nn.Module):
    def __init__(self,node_embedding_dim,edge_embedding_dim,graph_dim = 50,prop_steps = 2):
        super(Model,self).__init__()

        self.R = Graph_Representation(node_embedding_dim,edge_embedding_dim,graph_dim,prop_steps)
        self.f_addnode = f_addnode(graph_dim)
        self.f_addedge = f_add_edge(node_embedding_dim,graph_dim)
        self.f_nodes = f_nodes(node_embedding_dim)
        self.new_node = new_node(node_embedding_dim,edge_embedding_dim,graph_dim,prop_steps)
        # self.new_edge = new_edge(node_embedding_dim,edge_embedding_dim,graph_dim,prop_steps)

    def forward(self,batch,sequence,verbose = False):

        # dimension of h_v = (|V|, *)
        '''
        C(1)=C(2)=C(3)
        [(1,C), (3,C), (2,C)]---[[],[],(1,2,=), (3,2,=)]
        Node_type->index
        '''
        node_ordering = sequence[0]
        edge_ordering = sequence[1]

        log_p = torch.zeros((1),dtype=torch.float).to(device)
        # log p(G,π)

        canonical_node_ordering = dict()
        canonical_node_ordering[node_ordering[0][0]] = 0
        i = 1

        seed_graph = Data()
        
        seed_graph.x = torch.reshape(batch.x[ node_ordering[0][0] ],(1,128))
        seed_graph.edge_attr = torch.zeros((0,47),dtype=torch.float)
        seed_graph.edge_index = torch.zeros((2,0),dtype=torch.long)
        seed_graph = seed_graph.to(device)


        for edges,node in zip(edge_ordering[1:],node_ordering[1:]):

            h_G = self.R(seed_graph)

            node_type = self.f_addnode(h_G)
            if verbose:
                print("Add Node with type")
                print(node_type.cpu().data.numpy(),node[1])   
            log_p += torch.log( node_type[node[1]-1] )

            new_node_embedding = self.new_node(seed_graph,node_type[:-1])
            canonical_node_ordering[node[0]] = i
            i+=1
            # add new node to graph
            temp = seed_graph.x
            # batch.x = torch.cat([seed_graph.x,torch.reshape(new_node_embedding,(1,new_node_embedding.size()[0]))],dim=0)
            seed_graph.x = torch.cat([seed_graph.x, torch.reshape( batch.x[node[0]],(1 ,batch.x.size()[1] ) ) ],dim=0)
            # if len(edges) == 0:
            #     add_edge = self.f_addedge(h_G,new_node_embedding)
            #     if verbose:
            #         print("Add Edge or not")
            #         print(add_edge.data.numpy(),"no")
            #     log_p += torch.log(add_edge[1])
            for edge in edges:
                v = canonical_node_ordering[ node[0] ]
                if edge[0] == node[0]:
                    u = canonical_node_ordering[ edge[1]]
                else:
                    u = canonical_node_ordering[ edge[0] ]
                add_edge = self.f_addedge(h_G,new_node_embedding)
                if verbose:
                    print("Add Edge or not")
                    print(add_edge.cpu().data.numpy(),"yes")
                scores = self.f_nodes(temp,new_node_embedding)
                if verbose:
                    print("Which type of edge?")
                    print(scores.cpu().data.numpy(),"Node:",u,"Type:",edge[2])
                log_p += torch.log(add_edge[0]) + torch.log( scores[ u, edge[2] ] )   
                # add new edges to graph

                # seed_graph.edge_index = torch.cat([seed_graph.edge_index,torch.reshape(torch.tensor([u,v]).to(device),(2,1))],dim=1)
                # seed_graph.edge_attr = torch.cat([seed_graph.edge_attr,torch.zeros(1,47).to(device)],dim=0)
                # new_edge_embedding = self.new_edge(batch,scores[u])

                # batch.edge_attr = torch.cat([batch.edge_attr,new_edge_embedding],dim=0)
                # new_edge_type = torch.zeros((20)).to(device)
                # new_edge_type[edge[2]] = 1
                # batch.edge_attr[-1] = torch.cat([new_edge_type,new_edge_embedding],dim=0)
                edge_idx = 0
                for (i_,edge_) in enumerate(batch.edge_index.T):
                    if (edge_[0] == u and edge_[1] == v) or (edge_[0] == v and edge_[1] == u):
                        edge_idx = i_
                        break
                seed_graph.edge_index = torch.cat([seed_graph.edge_index,torch.reshape(batch.edge_index[:,edge_idx],(2,1))],dim=1)
                seed_graph.edge_attr = torch.cat([seed_graph.edge_attr,torch.reshape(batch.edge_attr[edge_idx],(1,47))],dim=0)
            add_edge = self.f_addedge(h_G,new_node_embedding)
            if verbose:
                print("Add Edge or not")
                print(add_edge.cpu().data.numpy(),"no")
            log_p += torch.log(add_edge[1]) 
            # print(batch)
        h_G = self.R(seed_graph)
        node_type = self.f_addnode(h_G)
        if verbose:
            print("Don't add node and STOP")
            print(node_type.cpu().data.numpy())
        log_p += torch.log( node_type[-1] )
        return -1*log_p

def draw_graph(G):
    fig = plt.figure()
    elements = nx.get_node_attributes(G, name = "element")
    nx.draw(G, with_labels=True, labels = elements, pos=nx.spring_layout(G))
    rect = plt.Rectangle((0, 0), 1, 1, fill=False, color="k", lw=2, zorder=1000, transform=fig.transFigure, figure=fig)
    fig.patches.extend([rect])
    plt.tight_layout()
    plt.show()

def sample(model):
    G = nx.Graph()
    seed_graph = Data()
    
    seed_graph = nx_to_torch_geom(read_molecule_nx('C'))

    seed_graph.edge_attr = torch.zeros((0,47),dtype=torch.float)
    seed_graph.edge_index = torch.zeros((2,0),dtype=torch.long)
    G.add_node (0,atomic_num = 6, element = ele(6).symbol)
    print("--------Initial Graph------")
    print(seed_graph)
    draw_graph(G)
    

    seed_graph.to(device)

    h_G = model.R(seed_graph)

    node_type = model.f_addnode(h_G)

    num_nodes = 1

    node_type_choice = np.random.choice(a = list(range(115)),p = node_type.cpu().data.numpy())

    while num_nodes < 50 and node_type_choice != 114:

        G.add_node(num_nodes,atomic_num = node_type_choice+1,zaib = ele(int(node_type_choice)+1).symbol)

        new_node_embedding = model.new_node(seed_graph,node_type[:-1])

        # add new node to graph
        temp = seed_graph.x
        seed_graph.x = torch.cat([seed_graph.x,torch.reshape(new_node_embedding,(1,new_node_embedding.size()[0]))],dim=0)
        num_nodes += 1

        print("--------Adding Node------")
        print(seed_graph)
        draw_graph(G)

        # decide whether to add edge or not.
        add_edge = model.f_addedge(h_G,new_node_embedding)
        add_edge_choice = np.random.choice(a = list(range(2)),p = add_edge.cpu().data.numpy())
        edges_added = set()
        while add_edge_choice != 1:
            scores = model.f_nodes(temp,new_node_embedding)

            # print(np.sum(scores.cpu().data.numpy().flatten()))
            edge_type_choice = np.random.choice(a = list(range(20*(num_nodes-1))),p = scores.cpu().data.numpy().flatten()/np.sum(scores.cpu().data.numpy().flatten()))

            node_type_choice = edge_type_choice // 20
            edge_type_choice = edge_type_choice % 20
            if node_type_choice in edges_added:
                break

            edges_added.add(node_type_choice)

            # add edge to graph
            # G.add_edge( num_nodes,node_type_choice , bond_type = rdchem.BondType.values[edge_type_choice+1]  )
            G.add_edge( num_nodes-1,int(node_type_choice) )
            
            seed_graph.edge_index = torch.cat([seed_graph.edge_index,torch.reshape(torch.tensor([node_type_choice,num_nodes-1]).to(device),(2,1))],dim=1)
            edge_data = torch.zeros(1,20).to(device)
            edge_data[ 0, edge_type_choice ] = 1
            edge_data = torch.cat([ edge_data,torch.zeros(1,27).to(device)],dim = 1)
            seed_graph.edge_attr = torch.cat([seed_graph.edge_attr,edge_data],dim=0)
            # print(seed_graph.x)
            # print(seed_graph.edge_attr)
            # print(seed_graph.edge_index)
            print("--------Adding Edge------")
            print(seed_graph)
            draw_graph(G)


            add_edge = model.f_addedge(h_G,new_node_embedding)
            add_edge_choice = np.random.choice(a = list(range(2)),p = add_edge.cpu().data.numpy())

        # nx.draw(G)
        h_G = model.R(seed_graph)

        node_type = model.f_addnode(h_G)
        node_type_choice = np.random.choice(a = list(range(115)),p = node_type.cpu().data.numpy())
    return (G,seed_graph)

def sample_MAP(model):
    G = nx.Graph()
    seed_graph = Data()
    
    seed_graph = nx_to_torch_geom(read_molecule_nx('C'))

    seed_graph.edge_attr = torch.zeros((0,47),dtype=torch.float)
    seed_graph.edge_index = torch.zeros((2,0),dtype=torch.long)
    G.add_node (0,atomic_num = 6, element = ele(6).symbol)
    print("--------Initial Graph------")
    print(seed_graph)
    draw_graph(G)
    

    seed_graph.to(device)

    h_G = model.R(seed_graph)

    node_type = model.f_addnode(h_G)

    num_nodes = 1

    node_type_choice = np.argmax(node_type.cpu().data.numpy())

    while num_nodes < 40 and node_type_choice != 114:

        G.add_node(num_nodes,atomic_num = node_type_choice+1,zaib = ele(int(node_type_choice)+1).symbol)

        new_node_embedding = model.new_node(seed_graph,node_type[:-1])

        # add new node to graph
        temp = seed_graph.x
        seed_graph.x = torch.cat([seed_graph.x,torch.reshape(new_node_embedding,(1,new_node_embedding.size()[0]))],dim=0)
        num_nodes += 1

        print("--------Adding Node------")
        print(seed_graph)
        draw_graph(G)

        # decide whether to add edge or not.
        add_edge = model.f_addedge(h_G,new_node_embedding)
        add_edge_choice = np.random.choice(a = list(range(2)),p = add_edge.cpu().data.numpy())
        edges_added = set()
        while add_edge_choice != 1:
            scores = model.f_nodes(temp,new_node_embedding)

            # print(np.sum(scores.cpu().data.numpy().flatten()))
            edge_type_choice = np.random.choice(a = list(range(20*(num_nodes-1))),p = scores.cpu().data.numpy().flatten()/np.sum(scores.cpu().data.numpy().flatten()))

            node_type_choice = edge_type_choice // 20
            edge_type_choice = edge_type_choice % 20
            if node_type_choice in edges_added:
                break

            edges_added.add(node_type_choice)

            # add edge to graph
            # G.add_edge( num_nodes,node_type_choice , bond_type = rdchem.BondType.values[edge_type_choice+1]  )
            G.add_edge( num_nodes-1,int(node_type_choice) )
            
            seed_graph.edge_index = torch.cat([seed_graph.edge_index,torch.reshape(torch.tensor([node_type_choice,num_nodes-1]).to(device),(2,1))],dim=1)
            edge_data = torch.zeros(1,20).to(device)
            edge_data[ 0, edge_type_choice ] = 1
            edge_data = torch.cat([ edge_data,torch.zeros(1,27).to(device)],dim = 1)
            seed_graph.edge_attr = torch.cat([seed_graph.edge_attr,edge_data],dim=0)
            # print(seed_graph.x)
            # print(seed_graph.edge_attr)
            # print(seed_graph.edge_index)
            print("--------Adding Edge------")
            print(seed_graph)
            draw_graph(G)


            add_edge = model.f_addedge(h_G,new_node_embedding)
            add_edge_choice = np.random.choice(a = list(range(2)),p = add_edge.cpu().data.numpy())

        # nx.draw(G)
        h_G = model.R(seed_graph)

        node_type = model.f_addnode(h_G)
        node_type_choice = np.argmax(node_type.cpu().data.numpy())
    return (G,seed_graph)