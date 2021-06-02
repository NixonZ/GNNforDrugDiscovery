import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.nn import MessagePassing

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
        self.MPN_list = [ MPN(node_embedding_dim,edge_embedding_dim) for _ in range(prop_steps) ]
        self.LSTM_list = [ nn.LSTMCell(node_embedding_dim,node_embedding_dim) for _ in range(prop_steps) ]

        # Learning Graph representation Layers
        self.gm = nn.Linear(node_embedding_dim,graph_dim)
        self.fm = nn.Linear(node_embedding_dim,graph_dim)

        # Structure building decisions
        
        self.fae = nn.Linear(node_embedding_dim+graph_dim,2)

    def forward(self,batch):
        '''
        batch : Batch
        '''
        c = torch.zeros(batch.x.size())
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
        
        self.fan = nn.Linear(graph_dim,114)

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
        
        self.fs = nn.Linear(2*node_embedding_dim,21)
    
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

class new_edge(nn.Module):

    def __init__(self,node_embedding_dim,edge_embedding_dim,graph_dim,prop_steps):
        super(new_edge,self).__init__()

        self.R_init = Graph_Representation(node_embedding_dim,edge_embedding_dim,graph_dim,prop_steps)
        self.f_init = nn.Linear(graph_dim+21,edge_embedding_dim)

    def forward(self,batch,x_uv):
        x = torch.cat([self.R_init(batch),x_uv],dim=0)
        x = self.f_init(x)
        return x


'''
G = (V,E)
Step -by step Generation
We have to learn the probability that our model predicts this graph
'''

class Model(nn.Module):
    def __init__(self,node_embedding_dim,edge_embedding_dim,graph_dim = 50,prop_steps = 2):
        super(Model,self).__init__()

        self.R = Graph_Representation(node_embedding_dim,edge_embedding_dim,graph_dim,prop_steps)
        self.f_addnode = f_addnode(graph_dim)
        self.f_addedge = f_add_edge(node_embedding_dim,graph_dim)
        self.f_nodes = f_nodes(node_embedding_dim)
        self.new_node = new_node(node_embedding_dim,edge_embedding_dim,graph_dim,prop_steps)
        self.new_edge = new_edge(node_embedding_dim,edge_embedding_dim,graph_dim,prop_steps)

    def forward(self,batch, sequence):

        h_G = self.R(batch)
        # dimension of h_v = (|V|, *)
        '''
        C(1)=C(2)=C(3)
        [(1,C), (3,C), (2,C)]---[[],[],(1,2,=), (3,2,=)]
        Node_type->index
        '''
        node_ordering = sequence[0]
        edge_ordering = sequence[1]
        log_p = 0.0 # log p(G,π)
        for edges,node in zip(edge_ordering,node_ordering):
            node_type = self.f_addnode(h_G)
            log_p += torch.log( node_type[node[1]] )
            
            new_node_embedding = self.new_node(batch,node_type)

            # add new node to graph
            temp = batch.x
            batch.x = torch.cat([batch.x,torch.reshape(new_node_embedding,(1,new_node_embedding.size()[0]))],dim=0)
            
            for edge in edges:
                if len(edge) == 0:
                    add_edge = self.f_addedge(h_G,new_node_embedding)
                    log_p += torch.log(add_edge[1])
                    continue
                v = node[0]
                if edge[0] == v:
                    u = edge[1]
                else:
                    u = edge[0]
                add_edge = self.f_addedge(h_G,new_node_embedding)
                scores = self.f_nodes(temp,new_node_embedding)
        
                log_p += torch.log(add_edge[0]) + torch.log( scores[ u, edge[2] ] )

                # add new edges to graph
                batch.edge_index = torch.cat([batch.edge_index,torch.reshape(torch.tensor([u,v]),(2,1))],dim=1)
                batch.edge_attr = torch.cat([batch.edge_attr,torch.zeros(1,47)],dim=0)
                new_edge_embedding = self.new_edge(batch,scores[u])

                # batch.edge_attr = torch.cat([batch.edge_attr,new_edge_embedding],dim=0)
                batch.edge_attr[-1] = new_edge_embedding
            print(batch)
        node_type = self.f_addnode(h_G)
        log_p += torch.log( node_type[-1] )
        return log_p

