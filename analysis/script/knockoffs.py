import sys
import pdb
import pyreadr
import numpy as np
import pandas as pd

sys.path.append("/home/dloewe/projects/aecg_survival_lbbb/lib/deepknockoffs/DeepKnockoffs/")

from DeepKnockoffs import KnockoffMachine
from DeepKnockoffs import GaussianKnockoffs

r_data = pyreadr.read_r("../data/derived_data/train_data.rds")

def scale(x, center=True, scale=True):
    """Scale data using centering and/or unit variance"""

    x_scale = x[:]

    if center: 
        mean = x.mean()
        x_scale = x_scale - mean

    if scale: 
        sd   = x.std()
        x_scale = x_scale/sd

    return x_scale


x_train = r_data[None]
p = x_train.shape[1]
n = x_train.shape[0]

# Use second order knockoffs to get estimate of pairwise correlations

SigmaHat = np.cov(x_train, rowvar=False)

second_order = GaussianKnockoffs(SigmaHat, method = "equi", mu = np.mean(x_train, 0), tol = 1e-10)

corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)

x_train = x_train.apply(scale, axis = 1)

# Set the parameters for training deep knockoffs
pars = dict()
# Number of epochs
pars['epochs'] = 5
# Number of iterations over the full data per epoch
pars['epoch_length'] = 10
# Data type, either "continuous" or "binary"
pars['family'] = 'continuous'
# Dimensions of data
pars['p'] = p
# Size of test set
pars['test_size'] = 0
# Batch size
pars['batch_size'] = int(0.5*n)
# Learning rate
pars['lr'] = 0.001
# When to decrease learning rate (unused when equal to number of epochs)
pars['lr_milestones'] = [pars['epochs']]
# Width of the network (number of layers is fixed to 6)
pars['dim_h'] = int(10*p)
# Penalty for the MMD distance
pars['GAMMA'] = 1
# Penalty encouraging second-order knockoffs
pars['LAMBDA'] = 1
# Decorrelation penalty hyperparameter
pars['DELTA'] = 1
# Target pairwise correlations between variables and knockoffs
pars['target_corr'] = corr_g
# Kernel widths for the MMD measure (uniform weights)
pars['alphas'] = [1., 2., 4., 8., 16., 32., 64., 128.]

# Where the machine is stored
checkpoint_name = "../models/deepmodel"

# test to exclude 51

x_train = x_train.to_numpy()

# Initialize the machine
machine = KnockoffMachine(pars, checkpoint_name)

# Train the machine
#pdb.set_trace()
machine.train(x_train)

# Generate deep knockoffs
xk_train = machine.generate(x_train)

# Save knockoffs
pyreadr.write_rds("../data/derived_data/knockoffs.rds", pd.DataFrame(xk_train))
