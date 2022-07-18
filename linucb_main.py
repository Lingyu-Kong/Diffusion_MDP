import config.LinUCB_config as config
import torch
from LinUCB.agent import Agent
import wandb
import numpy as np

wandb.login()
wandb.init(project="Diffusion MDP_N", entity="kly20")

torch.set_printoptions(threshold=np.inf)

shared_params = config.shared_params

wandb.config = config.wandb_config

if __name__=="__main__":
    agent=Agent(**config.agent_params)
    agent.train(
        shared_params["num_steps"],
        shared_params["T"],
        shared_params["state_batch_size"],
        shared_params["action_batch_size"],)
    agent.actor.save_model("./model_save/actor_final_single_minibatch.pt")
    agent.critic.save_model("./model_save/critic_final_single_minibatch.pt")
    # agent.actor.load_model("./model_save/actor_final_single_minibatch.pt")
    # agent.critic.load_model("./model_save/critic_final_single_minibatch.pt")
    # agent.test(20,shared_params["state_batch_size"])