import torch
import torch.nn.functional as F
import time
import wandb
import matplotlib.pyplot as plt
import numpy as np

from LinUCB.actor import Actor
from LinUCB.critic import Critic
from LinUCB.sampler import Sampler
from model.metalayer import LinLayer
from utils.tensor_utils import to_tensor
from bfgs.bfgs_utils import compute,batch_compute


class Agent(object):
    def __init__(
        self,
        actor_params:dict,
        critic_params:dict,
        sampler_params:dict,
        linlayer_params:dict,
        temperature:float,
        cuda:bool,
    ):
        self.actor = Actor(**actor_params)
        self.critic = Critic(**critic_params)
        self.sampler = Sampler(**sampler_params)
        self.linlayer_params=linlayer_params
        self.temperature=temperature
        self.device=torch.device("cuda" if cuda else "cpu")

    def train(self,num_steps,T,state_batch_size,action_batch_size):
        for current_step in range(num_steps):
            start_time=time.time()
            state_batch=self.sampler.state_sample(state_batch_size)
            for state_index in range(state_batch.shape[0]):
                current_state=state_batch[state_index,:,:]
                theta_s=LinLayer(**self.linlayer_params)
                Sigma_s=torch.eye(state_batch.shape[1]*state_batch.shape[2]).to(self.device)
                A_s=torch.zeros(T,state_batch.shape[1],state_batch.shape[2]).to(self.device)
                reward_s=torch.zeros(T).to(self.device)
                for t in range(T):
                    action_batch=to_tensor(self.sampler.action_sample(action_batch_size))
                    s_prime=torch.repeat_interleave(to_tensor(current_state).unsqueeze(0),action_batch_size,dim=0)+action_batch
                    batch_reward=np.subtract([compute(current_state.tolist())]*action_batch_size,batch_compute(s_prime,action_batch_size))
                    action_flat_batch=action_batch.view(action_batch_size,-1)
                    kernel=torch.sqrt(torch.mm(action_flat_batch,torch.mm(torch.inverse(Sigma_s),action_flat_batch.transpose(0,1))))
                    kernel=torch.diag(kernel)
                    ucb=to_tensor(batch_reward)+self.temperature*kernel
                    chosen_index=torch.argmax(ucb)
                    chosen_action=action_batch[chosen_index,:]
                    reward=compute(current_state.tolist())-compute((to_tensor(current_state)+chosen_action).tolist())
                    # theta_s_loss=torch.mean(reward-theta_s(action_flat_batch[chosen_index].unsqueeze(0)).squeeze(0))
                    # theta_s.optimizer.zero_grad()
                    # theta_s_loss.backward()
                    # theta_s.optimizer.step()
                    A_s[t,:,:]=chosen_action
                    reward_s[t]=reward
                    # plt.figure()
                    # plt.plot(batch_reward.tolist(),label="reward",color="red")
                    # plt.plot((ucb).tolist(),label="ucb",color="blue")
                    # plt.plot((-theta_s(action_flat_batch)).squeeze(-1).tolist(),label="theta_s",color="green")
                    # plt.legend()
                    # wandb.log({"reward_"+str(t):plt})

                ## check A_s
                if current_step%20==0:
                    origin_energy=[compute(current_state.tolist())]*T
                    plt.figure()
                    plt.plot(origin_energy,label="origin_energy",color="red")
                    plt.plot(reward_s.tolist(),label="reward_s",color="blue")
                    plt.legend()
                    wandb.log({"check A_s "+str(current_step):plt})
                ## update critic
                s_prime=torch.repeat_interleave(to_tensor(current_state).unsqueeze(0),T,dim=0)+A_s
                critic_loss=F.mse_loss(self.critic.evaluate(s_prime).squeeze(-1),reward_s)
                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                self.critic.optimizer.step()
                ## update actor
                log_prob,_,_=self.actor(torch.repeat_interleave(to_tensor(current_state).unsqueeze(0),T,dim=0),A_s)
                actor_loss=torch.mean(-log_prob)
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
            end_time=time.time()
            ## log
            wandb.log({"critic_loss":critic_loss.item(),
                    "actor_loss":actor_loss.item()})
            print("step:",current_step,"     time cost:",(end_time-start_time))
            print("===============================================================")

    def test(self,num_steps,state_batch_size):
        for current_step in range(num_steps):
            start_time=time.time()
            state_batch=to_tensor(self.sampler.state_sample(state_batch_size))
            origin_energy=batch_compute(state_batch,state_batch_size)
            _,action_batch,_=self.actor.sample_action(state_batch)
            action_batch=action_batch.view(state_batch.shape[0],state_batch.shape[1],3)
            state_prime_batch=action_batch+state_batch
            print(state_batch[0])
            print(action_batch[0])
            energy_prime=batch_compute(state_prime_batch,state_batch_size)
            plt.figure()
            plt.plot(origin_energy,label="origin_energy",color="red")
            plt.plot(energy_prime,label="energy_prime",color="blue")
            plt.legend()
            wandb.log({"test_"+str(current_step):plt})
            end_time=time.time()
            print("step:",current_step,"     time cost:",(end_time-start_time))
            print("===============================================================")









        