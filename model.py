import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Message_passing(nn.Module):

    def __init__(self,node_embedding_dim,edge_embedding_dim,agg_message_dim):
        super(Message_passing,self).__init__()
        self.linear1 = nn.Linear(node_embedding_dim*2+edge_embedding_dim,agg_message_dim)

    def forward(self,h_u,h_v,x_uv):
        '''
        node_embeddding_u + node_embedding_v + edge_embedding_uv
        '''
        x = torch.cat([h_u,h_v,x_uv])
        x = self.linear1.forward(x)
        return

class Model(nn.Module):

    def __init__(self,node_embedding_dim,edge_embedding_dim,agg_message_dim,hidden_dim = 50,prop_steps = 100,batch_size = 64):
        super(Model,self).__init__()

        self.fe = Message_passing(node_embedding_dim,edge_embedding_dim,agg_message_dim)

    def forward(self,x):
        '''
        x : batch_size,graph
        '''