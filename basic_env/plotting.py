import numpy as np
from matplotlib import pyplot as plt

def return_plot(plot_name=None, dof = 4,**kwargs):
    
    
    sq = int(np.ceil(np.sqrt(dof)))
    fig, axes = plt.subplots(sq,sq,figsize=(16, 16)) 
    
    for key, value in kwargs.items():
        value=value[:,:dof]

        for idx in range(dof):
            if sq!=1:
                row = int(idx/sq)
                col = int(idx%sq)
                axes[row][col].plot(value[:,idx],label=key)
                axes[row][col].legend()
            else:
                axes.plot(value[:,idx],label=key)
                axes.legend()
        
    if plot_name is not None:
        plt.savefig("plots/"+plot_name, dpi=100)    
        
    return fig

def plot_values(plot_name=None, dof = 4,dont_show = False, **kwargs):
    
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
    elif dont_show==False:
        plt.show()
