import matplotlib.pyplot as plt
import numpy as np
import model_diagnostics

# import imp
# imp.reload(model_diagnostics)

colors = ('#284E60','#E1A730','#D95980','#C3B1E1','#351F27','#A9C961')
clr_shash = colors[0]
clr_bnn   = colors[3]
clr_truth = 'dimgray'



FS = 16
### for white background...
plt.rc('text',usetex=True)
plt.rc('font',**{'family':'sans-serif','sans-serif':['Avant Garde']}) 
plt.rc('savefig',facecolor='white')
plt.rc('axes',facecolor='white')
plt.rc('axes',labelcolor='dimgrey')
plt.rc('axes',labelcolor='dimgrey')
plt.rc('xtick',color='dimgrey')
plt.rc('ytick',color='dimgrey')

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 5))
        else:
            spine.set_color('none')  
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        ax.yaxis.set_ticks([])
    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
            ax.xaxis.set_ticks([])  

    
    
def plot_pits(ax, x_val, onehot_val, model_shash, shash_cpd):
    plt.sca(ax)      
    
    # shash pit
    bins, hist_shash, D_shash, EDp_shash = model_diagnostics.compute_pit('shash',onehot_val, x_data=x_val,model_shash=model_shash)
    bins_inc = bins[1]-bins[0]

    if bnn_cpd is not None:
        bin_add = bins_inc/6+bins_inc/6 
        bin_width = bins_inc/3
    else:
        bin_add = bins_inc/2
        bin_width = bins_inc*.98
    plt.bar(hist_shash[1][:-1] + bin_add,
             hist_shash[0],
             width=bin_width,
             color=clr_shash,
             label='SHASH',
            )
    
    # make the figure pretty
    plt.axhline(y=.1, 
                linestyle='--',
                color='dimgray', 
                linewidth=2.,
               )
    plt.ylim(0,.2)
    plt.xticks(bins,np.around(bins,1))
    ax = plt.gca()
    yticks = np.around(np.arange(0,.25,.05),2)
    plt.yticks(yticks,yticks)
    
    plt.text(0.,np.max(yticks)*.99,
             'SHASH D: ' + str(np.round(D_shash,4)) + ' (' + str(np.round(EDp_shash,3)) +  ')', 
             color=clr_shash,     
             verticalalignment='top',
             fontsize=12)


    plt.xlabel('probability integral transform')
    plt.ylabel('probability')
    plt.legend(loc=1)
    plt.title('PIT histogram comparison', fontsize=FS, color='k')
    
    
