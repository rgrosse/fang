import gnumpy as gnp
import re
import os

import config
import datasets
import rbm_training
from utils import storage

def outputs_dir(name):
    return os.path.join(config.OUTPUTS_DIR, 'small_rbms', name)

def rbm_file(name):
    return os.path.join(outputs_dir(name), 'rbm.pk')

    

def get_params(name):
    m = re.match('subsampled_mnist/([^/_]*)$', name)
    if m:
        lrate = float(m.group(1))
        return {'lrate': lrate,
                'num_steps': 100000,
                'name': name,
                }

    raise RuntimeError('Unknown experiment: {}'.format(name))


def run(params):
    if isinstance(params, str):
        params = get_params(params)

    v = gnp.garray(datasets.SubsampledMNISTInfo.load().as_matrix())
    v = 0.999 * v + 0.001 * 0.5

    tparams = rbm_training.TrainingParams.defaults('pcd')
    rbm, _ = rbm_training.train_rbm(v, 20, tparams, show_progress=True)

    storage.dump(rbm, rbm_file(params['name']))



