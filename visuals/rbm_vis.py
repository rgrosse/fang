import numpy as np
import pylab


import misc


def show_reconstructions(rbm, inputs, data_info, display=True, figname='reconstructions', figtitle='reconstructions'):
    reconst = rbm.reconstruct(inputs)
    imgs = [reconst[j, :].reshape(data_info.shape) for j in range(reconst.shape[0])]
    if display:
        pylab.figure(figname)
        misc.display(misc.pack(imgs))
        pylab.title(figtitle)
    return misc.norm01(misc.pack(imgs))

def show_hidden_activations(rbm, inputs, display=True, figname='hidden activations', figtitle='hidden activations'):
    hid = rbm.get_hidden(inputs)
    if display:
        pylab.figure(figname)
        misc.display(hid)
        pylab.title(figtitle)
    return hid

def show_receptive_fields(rbm,data_info, display=True, figname='receptive fields', figtitle='receptive fields'):
    imgs = [rbm.weights[:, j].as_numpy_array().reshape((data_info.num_rows, data_info.num_cols))
            for j in range(rbm.nhid)]    
    if display:
        pylab.figure(figname)
        misc.display(misc.pack(imgs))
        pylab.title(figtitle)
    return misc.norm01(misc.pack(imgs))

def show_particles(rbm, state, dataset, display=True, figname='PCD particles', figtitle='PCD particles',
                   size=None):
    try:
        fantasy_vis = rbm.vis_expectations(state.h)
    except:
        fantasy_vis = state
        
    if size is None:
        size = (dataset.num_rows, dataset.num_cols)
    imgs = [fantasy_vis[j, :np.prod(size)].reshape(size).as_numpy_array()
            for j in range(fantasy_vis.shape[0])]
    visual = misc.norm01(misc.pack(imgs))
    if display:
        pylab.figure(figname)
        pylab.matshow(visual, cmap='gray', fignum=False)
        pylab.title(figtitle)
    return visual
    
