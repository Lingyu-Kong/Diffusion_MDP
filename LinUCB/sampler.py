import numpy as np
from bfgs.bfgs_utils import relax

class Sampler(object):
    def __init__(
        self,
        num_atoms:int,
        pos_scale:float,
        action_scale:float,
        max_relax_steps:int,
        threshold:float,
    ):
        self.num_atoms = num_atoms
        self.pos_scale = pos_scale
        self.action_scale = action_scale
        self.max_relax_steps = max_relax_steps
        self.threshold = threshold
    
    def state_sample(self,batch_size):
        state_batch=np.zeros((batch_size,self.num_atoms,3))
        for k in range(batch_size):
            pos=np.zeros((self.num_atoms,3))
            for i in range(self.num_atoms):
                if_continue=True
                while if_continue:
                    new_pos=np.random.rand(3)*2*self.pos_scale-self.pos_scale
                    if_continue=False
                    for j in range(i):
                        distance=np.linalg.norm(new_pos-pos[j],ord=2)
                        if distance<self.threshold:
                            if_continue=True
                            break
                pos[i,:]=new_pos
            _,_,pos=relax(pos,self.max_relax_steps)
            mutation=np.random.rand(self.num_atoms,3)*2*self.action_scale-self.action_scale
            state_batch[k,:,:]=np.subtract(pos,mutation)
        return state_batch
    
    def action_sample(self,batch_size):
        action_batch=np.zeros((batch_size,self.num_atoms,3))
        for i in range(batch_size):
            action_batch[i,:,:]=np.random.rand(self.num_atoms,3)*2*self.action_scale-self.action_scale
        return action_batch