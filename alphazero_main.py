import config.alphazero_config
import torch
from alphazero.agent import Agent
import wandb
import numpy as np

wandb.login()
wandb.init(project="MDP_N", entity="kly20")

torch.set_printoptions(threshold=np.inf)

shared_params = config.shared_params

wandb.config = config.wandb_config

if __name__=="__main__":
    agent=Agent(**config.alphazero_agent_params)
    agent.train(
        shared_params["num_steps"],
        shared_params["warmup_steps"],
        shared_params["batch_size"])
    agent.actor.save_model("./model_save/actor_final_single_minibatch.pt")
    # agent.critic.save_model("./model_save/critic_final_single_minibatch.pt")
    # agent.actor.load_model("./model_save/actor_final.pt")
    # agent.critic.load_model("./model_save/critic_final.pt")
    # agent.test(args.buffer_warmup_steps,200)