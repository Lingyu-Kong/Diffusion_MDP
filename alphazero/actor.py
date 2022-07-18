import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as Dist
from model.metalayer import MLPwoLastAct
from model.dmcg_nn import DMCG_NN_Pos_Fixed

"""
多分类定义的 actor
"""


class Actor(nn.Module):
    def __init__(self,gnn_params,num_atoms,action_choices,latent_size,mlp_hidden_size,mlp_layers,lr,cuda):
        super().__init__()
        self.gnn=DMCG_NN_Pos_Fixed(**gnn_params)     
        self.classify_net=MLPwoLastAct(
            input_size=latent_size*num_atoms,
            output_sizes=[mlp_hidden_size]*mlp_layers+[action_choices],
            use_layer_norm=False,
            activation=nn.ReLU,
            dropout=0.0,
            layernorm_before=False,
            use_bn=False,
        )
        self.final_act=nn.Softmax(dim=1)

        self.device=torch.device("cuda" if cuda else "cpu")
        self.classify_net.to(self.device)
        self.optimizer=optim.Adam(self.parameters(),lr=lr)
        self.weight_init(self.classify_net)
        self.weight_init(self.gnn)

    def forward(self,state_batch):
        """
        state_batch: [batch_size,num_atoms,3]
        node_attr: [batch_size*num_atoms,latent_size] -> [batch_size,num_atoms*latent_size]
        """
        node_attr,_,_=self.gnn(state_batch)
        node_attr=node_attr.view(state_batch.shape[0],-1)
        x=self.classify_net(node_attr)
        prob=self.final_act(x)
        return prob

    def save_model(self,path):
        torch.save(self.state_dict(),path)

    def load_model(self,path):
        self.load_state_dict(torch.load(path))

    def weight_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
        
        