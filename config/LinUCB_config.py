shared_params = {
    "num_steps": 1000,
    "decay_steps":50,
    "decay_rate":0.9,
    "pos_scale": 2.0,
    "num_atoms": 10,
    "state_batch_size": 8,
    "action_batch_size": 128,
    "T": 10,
    "threshold": 1,
    "action_scale":0.1,
    "max_relax_steps":200,
    "temperature":10,
}

wandb_config={
    "num_steps": shared_params["num_steps"],
    "num_atoms": shared_params["num_atoms"],
}

actor_gnn_params = {
    "cuda": True,
    "lr": 1e-4,
    "num_atoms":shared_params["num_atoms"],
    "mlp_hidden_size":512,
    "mlp_layers":2,
    "latent_size":256,
    "use_layer_norm":False,
    "num_message_passing_steps":12,
    "global_reducer":"sum",
    "node_reducer":"sum",
    "dropedge_rate":0.1,
    "dropnode_rate":0.1,
    "dropout":0.1,
    "layernorm_before":False,
    "use_bn":False,
    "cycle":1,
    "node_attn":True,
    "global_attn":True,
}

actor_params={
    "gnn_params":actor_gnn_params,
    "pos_scale":shared_params["pos_scale"],
    "latent_size":256,
    "mlp_hidden_size":512,
    "mlp_layers":2,
    "lr":1e-4,
    "cuda":True,
}

critic_gnn_params = {
    "cuda": True,
    "lr": 1e-4,
    "num_atoms":shared_params["num_atoms"],
    "mlp_hidden_size":512,
    "mlp_layers":2,
    "latent_size":256,
    "use_layer_norm":False,
    "num_message_passing_steps":12,
    "global_reducer":"sum",
    "node_reducer":"sum",
    "dropedge_rate":0.1,
    "dropnode_rate":0.1,
    "dropout":0.1,
    "layernorm_before":False,
    "use_bn":False,
    "cycle":1,
    "node_attn":True,
    "global_attn":True,
}

critic_params={
    "gnn_params":critic_gnn_params,
    "latent_size":256,
    "mlp_hidden_size":512,
    "mlp_layers":2,
    "lr":1e-4,
    "cuda":True,
}

sampler_parmas={
    "num_atoms":shared_params["num_atoms"],
    "pos_scale":shared_params["pos_scale"],
    "action_scale":shared_params["action_scale"],
    "max_relax_steps":shared_params["max_relax_steps"],
    "threshold":shared_params["threshold"],
}

linlayer_params={
    "in_dim":shared_params["num_atoms"]*3,
    "out_dim":1,
    "lr":1e-4,
    "cuda":True,
}

agent_params={
    "actor_params":actor_params,
    "critic_params":critic_params,
    "sampler_params":sampler_parmas,
    "linlayer_params":linlayer_params,
    "temperature":shared_params["temperature"],
    "cuda":True,
}