import numpy as np
from matplotlib import pyplot as plt

def plot_values(plot_name=None, dof = 4,**kwargs):
    
    plt.figure(figsize=(16, 16)) 
    sq = int(np.ceil(np.sqrt(dof)))

    for key, value in kwargs.items():
        value=value[:,:dof]

        for idx in range(dof):
            # print(idx)
            plt.subplot(sq,sq,idx+1)
            plt.plot(value[:,idx],label=key)
            plt.legend()
        
    if plot_name is not None:
        plt.savefig("plots/"+plot_name, dpi=100)    
    
    plt.show()
    