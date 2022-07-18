shared_params = {
    "num_steps": 1000,
    "decay_steps":200,
    "decay_rate":0.9,
    "pos_scale": 2.0,
    "num_atoms": 10,
    "batch_size": 64,
    "threshold": 1,
    "action_scale":0.1,
    "unit_length":0.01,
    "warmup_steps":200,
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
    "num_atoms":shared_params["num_atoms"],
    "action_choices":int((shared_params["action_scale"]*2)/shared_params["unit_length"]*3*shared_params["num_atoms"]),
    "latent_size":actor_gnn_params["latent_size"],
    "mlp_hidden_size":actor_gnn_params["mlp_hidden_size"],
    "mlp_layers":actor_gnn_params["mlp_layers"],
    "lr":1e-4,
    "cuda":True,
}

replay_buffer_params={
    "buffer_size":2000,
    "num_atoms":shared_params["num_atoms"],
    "action_choices":int((shared_params["action_scale"]*2)/shared_params["unit_length"]*3*shared_params["num_atoms"]),
}

agent_params={
    "actor_params":actor_params,
    "replay_buffer_params":replay_buffer_params,
    "num_atoms":shared_params["num_atoms"],
    "pos_scale":shared_params["pos_scale"],
    "threshold":shared_params["threshold"],
    "action_scale":shared_params["action_scale"],
    "unit_length":shared_params["unit_length"],
    "max_relax_steps":200,
    "mcts_times":1600,
}

wandb_config = {
    "num_atoms":shared_params["num_atoms"]
}