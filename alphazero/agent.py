from cProfile import label
import sys 
sys.path.append("..") 

import torch
import numpy as np
import time
import wandb
import matplotlib.pyplot as plt

from alphazero.actor import Actor
from alphazero.critic import Critic
from utils.random_utils import get_random_conform,get_random_mutation
from bfgs.bfgs_utils import relax,compute
from utils.tensor_utils import to_tensor

class Agent(object):
    def __init__(
        self,
        actor_params:dict,
        # critic_params:dict,
        replay_buffer_params:dict,
        num_atoms:int,
        pos_scale:float,
        threshold:float,
        action_scale:float,
        unit_length:float,
        max_relax_steps:int,
        mcts_times:int,
    ):
        self.actor=Actor(**actor_params)
        # self.critic=Critic(**critic_params)
        self.replay_buffer=ReplayBuffer(**replay_buffer_params)
        self.num_atoms=num_atoms
        self.pos_scale=pos_scale
        self.threshold=threshold
        self.action_scale=action_scale
        self.unit_length=unit_length
        self.max_relax_steps=max_relax_steps
        self.mcts_times=mcts_times

    def train(self,num_steps,warmup_steps,batch_size):
        for i in range(num_steps):
            start_time=time.time()
            state,action_prob,action_value=self.self_play_single_sample()
            self.replay_buffer.store(state,action_prob,action_value)
            if i>=warmup_steps:
                self.update_one_iter(batch_size,i)
            end_time=time.time()
            print("step:"+i.__str__()+"   finished, time_cost:"+(end_time-start_time).__str__())
            print("=================================================================")

    def update_one_iter(self,batch_size,i):
        state_batch,action_prob_batch,action_value_batch=self.replay_buffer.read(batch_size)
        predicted_prob=self.actor(to_tensor(state_batch))

        # log
        if i%100==99:
            plt.figure()
            axis1=plt.subplot()
            axis2=axis1.twinx()
            axis1.plot(predicted_prob[0].tolist(),label="predicted_prob",color='red')
            axis1.plot(action_prob_batch[0].tolist(),label="action_prob",color='blue')
            axis2.plot(action_value_batch[0].tolist(),label="action_value",color='green')
            wandb.log({"performance"+i.__str__():plt})

        # predicted_value=self.critic(to_tensor(state_batch))
        actor_loss=torch.mean(torch.sum(-predicted_prob*torch.log(to_tensor(action_prob_batch)),dim=-1))
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()


    def self_play_batch_sample(self,batch_size):
        pass

    def self_play_single_sample(self):
        random_pos=get_random_conform(self.num_atoms,self.pos_scale,self.threshold)
        _,_,local_minimal=relax(random_pos.tolist(),self.max_relax_steps)
        mutation=get_random_mutation(self.num_atoms,self.action_scale)
        root_state=np.add(local_minimal,mutation)
        action_prob,action_value=self.mcts_for_action(root_state)
        return root_state,action_prob,action_value


    def mcts_for_action(self,root_state):
        action_choices_num=int((self.action_scale*2)/self.unit_length*3*self.num_atoms)
        N_root=1
        N=np.ones(action_choices_num)
        P=self.actor(to_tensor(root_state).unsqueeze(0)).detach().squeeze(0).cpu().numpy()
        W=np.zeros(action_choices_num)
        for _ in range(self.mcts_times):
            ucb=0.1*np.divide(W,N)+(P*np.sqrt(2*np.log(N_root)/N))
            chosen_index=np.argmax(ucb)
            action=self.label_to_action(chosen_index,self.action_scale,self.unit_length)
            reward=compute(np.add(root_state,action).tolist())
            N[chosen_index]+=1
            W[chosen_index]+=reward
            N_root+=1
        prob=np.divide(N,N_root)
        return prob,np.divide(W,N)


    def label_to_action(self,label,action_scale,unit_length):
        b=(action_scale*2)/unit_length
        pos_index=int(label//b)
        atom_index=int(pos_index//3)
        pos_index=int(pos_index%3)
        mutation=-action_scale+(label%b)*unit_length
        action=np.zeros((self.num_atoms,3),dtype=np.float32)
        action[atom_index][pos_index]=mutation
        return action
        

class ReplayBuffer(object):
    def __init__(
        self,
        buffer_size:int,
        num_atoms:int,
        action_choices:int,
    ):
        self.buffer_size=buffer_size
        self.buffer_top=0
        self.state_buffer=np.zeros((buffer_size,num_atoms,3))
        self.action_prob_buffer=np.zeros((buffer_size,action_choices))
        self.action_value_buffer=np.zeros((buffer_size,action_choices))
    
    def store(self,state,action_prob,action_value):
        self.state_buffer[self.buffer_top%self.buffer_size,:,:]=state
        self.action_prob_buffer[self.buffer_top%self.buffer_size,:]=action_prob
        self.action_value_buffer[self.buffer_top%self.buffer_size,:]=action_value
        self.buffer_top=(self.buffer_top+1)%self.buffer_size

    def read(self,batch_size):
        choices=np.random.choice(min(self.buffer_size,self.buffer_top),batch_size)
        return self.state_buffer[choices,:,:],self.action_prob_buffer[choices,:],self.action_value_buffer[choices,:]