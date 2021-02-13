import numpy as np
import pickle
import jax,jax.numpy as jp
from jax import random
from jax.config import config
config.enable_omnistaging()
config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
from CNN_models import singleCNN,MPSR_singleCNN

# choose model here
which_model_will_you_run='MPSR_singleCNN'

#lattice is 10x10
L=10
K=9
k=int((K-1)/2)
site_number=int(L**2)

if which_model_will_you_run=='singleCNN':
    model=singleCNN()
    sign_rule_switch=0
    s_start = np.load('singleCNN/pre_hot_spin.npy')
    with open('singleCNN/modelchk_singleCNN','rb') as f:
        params_chk=pickle.load(f)
else:
    model=MPSR_singleCNN()
    sign_rule_switch=1
    s_start = np.load('MPSR_singleCNN/pre_hot_spin.npy')
    with open('MPSR_singleCNN/modelchk_MPSRsingleCNN','rb') as f:
        params_chk=pickle.load(f)

#determine params shape
key=random.PRNGKey(0)
s1=jp.array(np.random.normal(size=[1,L+K-1,L+K-1,1]),dtype=jp.float64)
s2=jp.array(np.random.normal(size=[1,L+K-1,L+K-1,1]),dtype=jp.float64)
s3=jp.array(np.random.normal(size=[1,L+K-1,L+K-1,1]),dtype=jp.float64)
s4=jp.array(np.random.normal(size=[1,L+K-1,L+K-1,1]),dtype=jp.float64)
params=model.init(key,s1,s2,s3,s4)
jax.tree_map(lambda x:x.shape,params)

# start mc process
from mc import MCMC
Es_list, spin_lattice=MCMC(1000,s_start,model,params_chk,sign_rule_switch)
with open('energy','a') as f:
    f.write('Model: {}, Energy per site: {}\n '.format(which_model_will_you_run,np.mean(Es_list)/400))
