import numpy as np

def get_random_conform(num_atoms,pos_scale,threshold):
    """
    Generate a random conformation of the atoms satisfying the constraints.
    """
    pos=np.zeros((num_atoms,3))
    for i in range(num_atoms):
        if_continue=True
        while if_continue:
            new_pos=np.random.rand(3)*2*pos_scale-pos_scale
            if_continue=False
            for j in range(i):
                distance=np.linalg.norm(new_pos-pos[j],ord=2)
                if distance<threshold:
                    if_continue=True
                    break
        pos[i,:]=new_pos
    return pos

def get_random_mutation(num_atoms,mutation_scale):
    """
    Generate a random mutation of the atoms.
    """
    mutation=np.zeros((num_atoms,3))
    mutation_index=np.random.randint(low=0,high=num_atoms)
    mutation[mutation_index,:]=np.random.rand(3)*2*mutation_scale-mutation_scale
    return mutation