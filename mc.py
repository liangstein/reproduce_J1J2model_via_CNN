import jax,jax.numpy as jp
from numba import njit
import numpy as np

#lattice is 10x10
L=10
K=9
k=int((K-1)/2)
site_number=int(L**2)

@njit
def sign_rule(spin_lattice,switch):
    s=0.5*(spin_lattice+1)
    if switch==0:
        return 1
    else:
        # J1=0
        #k=s[::2,:]
        #M=int(jp.sum(k))
        # J2=0
        s1=s[::2,::2]
        s2=s[1::2,1::2]
        M=np.sum(s1)+np.sum(s2)
        return (-1)**M


#PBC padding
@njit
def make_PBC_spin_lattice(spin_lattice):
    PBC=np.zeros(shape=(len(spin_lattice),L+K-1,L+K-1))
    k=int((K-1)/2)
    PBC[:,k:L+k,k:L+k]=spin_lattice[:,:,:]
    PBC[:,:k,:k]=spin_lattice[:,-k:,-k:]
    PBC[:,:k, -k:]=spin_lattice[:,-k:, :k]
    PBC[:,-k:, :k]=spin_lattice[:,:k, -k:]
    PBC[:,-k:, -k:]=spin_lattice[:,:k, :k]
    PBC[:,:k, k:L+k]=spin_lattice[:,-k:, :]
    PBC[:,-k:, k:L+k]=spin_lattice[:,:k, :]
    PBC[:,k:L+k, :k]=spin_lattice[:,:, -k:]
    PBC[:,k:L+k, -k:]=spin_lattice[:,:, :k]
    return PBC #[batchsize,L+K-1,L+K-1]



def WS_CALCULATE(spin_lattice, model, params, sign_rule_switch):
    s_1=make_PBC_spin_lattice(spin_lattice.reshape(1,L,L))
    spin_input_1=s_1.reshape((1, L+K-1, L+K-1,1))
    spin_input_2=np.rot90(spin_input_1,1,axes=(1,2))
    spin_input_3=np.rot90(spin_input_1,2,axes=(1,2))
    spin_input_4=np.rot90(spin_input_1,3,axes=(1,2))
    ws = model.apply(params,jp.array(spin_input_1),
                     jp.array(spin_input_2),
                     jp.array(spin_input_3),
                     jp.array(spin_input_4)).squeeze()*sign_rule(spin_lattice,sign_rule_switch)
    del params,model
    return ws



def calculate_sprime(spin_lattice,J2):
    E_s = 0
    propose_batch=[]
    s1length = 0
    # J1 interactions
    for i in range(0,L):
        for j in range(0,L):
            site_1 = [i, j]
            if i + 1 < L:
                site_2 = [i + 1, j]
            else:
                site_2 = [0, j]
            spin_1 = spin_lattice[site_1[0], site_1[1]]
            spin_2 = spin_lattice[site_2[0], site_2[1]]
            if spin_1!=spin_2:
                propose_batch.append([site_1,spin_2,site_2,spin_1])
                s1length += 1
                E_s -= 1
            else:
                E_s += 1
            if j + 1 < L:
                site_2 = [i, j + 1]
            else:
                site_2 = [i, 0]
            spin_1 = spin_lattice[site_1[0], site_1[1]]
            spin_2 = spin_lattice[site_2[0], site_2[1]]
            if spin_1!=spin_2:
                propose_batch.append([site_1, spin_2, site_2, spin_1])
                s1length += 1
                E_s -= 1
            else:
                E_s += 1
    # J2 interactions
    if J2!=0:
        for i in range(0,L):
            for j in range(0,L):
                site_1 = [i, j]
                if i + 1 < L and j + 1 < L:
                    site_2 = [i + 1, j + 1]
                elif i + 1 < L and j + 1 >= L:
                    site_2 = [i + 1, 0]
                elif i + 1 >= L and j + 1 < L:
                    site_2 = [0, j + 1]
                elif i + 1 >= L and j + 1 >= L:
                    site_2 = [0, 0]
                spin_1 = spin_lattice[site_1[0], site_1[1]]
                spin_2 = spin_lattice[site_2[0], site_2[1]]
                if spin_1!=spin_2:
                    propose_batch.append([site_1, spin_2, site_2, spin_1])
                    E_s -= J2
                else:
                    E_s += J2
                if i + 1 < L and j - 1 >= 0:
                    site_2 = [i + 1, j - 1]
                elif i + 1 < L and j - 1 < 0:
                    site_2 = [i + 1, -1]
                elif i + 1 >= L and j - 1 >= 0:
                    site_2 = [0, j - 1]
                elif i + 1 >= L and j - 1 < 0:
                    site_2 = [0, -1]
                spin_1 = spin_lattice[site_1[0], site_1[1]]
                spin_2 = spin_lattice[site_2[0], site_2[1]]
                if spin_1!=spin_2:
                    propose_batch.append([site_1, spin_2, site_2, spin_1])
                    E_s -= J2
                else:
                    E_s += J2
    return E_s,propose_batch,s1length



def make_s_prime(spin_lattice,propose_batch):
    s_prime_batch=np.zeros(shape=(len(propose_batch),L,L)) #[batchsize,L,L]
    s_prime_batch[:,:,:]=spin_lattice[:,:]
    for i,propose in enumerate(propose_batch):
        site_1, spin_2, site_2, spin_1=propose[0],propose[1],propose[2],propose[3]
        s_prime_batch[i,site_1[0],site_1[1]]=spin_2
        s_prime_batch[i,site_2[0], site_2[1]]=spin_1
    return s_prime_batch #[batchsize,L,L]


def calculate_local_energy(ws, spin_lattice, model, params, sample_option, sign_rule_switch):
    J2=0.5
    if sample_option==1:
        E_s, propose_batch, s1length = calculate_sprime(spin_lattice, J2)
        # computing the total non-diagonal Es elements
        if len(propose_batch) != 0:
            batchsize=len(propose_batch)
            s_prime=make_s_prime(spin_lattice,propose_batch) # [batchsize,L,L]
            s_prime_PBC=make_PBC_spin_lattice(s_prime) #[batchsize,L+K-1,L+K-1]
            spin_input_1=s_prime_PBC.reshape((batchsize,L+K-1,L+K-1,1))
            spin_input_2=np.rot90(spin_input_1,1,axes=(1,2))
            spin_input_3=np.rot90(spin_input_1,2,axes=(1,2))
            spin_input_4=np.rot90(spin_input_1,3,axes=(1,2))
            ws_1_batch=model.apply(params,jp.array(spin_input_1),
                     jp.array(spin_input_2),
                     jp.array(spin_input_3),
                     jp.array(spin_input_4)).squeeze()
            sign_batch=jp.array([sign_rule(s,sign_rule_switch) for s in s_prime])
            ws_1_batch = ws_1_batch*sign_batch
            E_s += (2 / ws) * (jp.sum(ws_1_batch[:s1length]) + J2 * jp.sum(ws_1_batch[s1length:]))
        del params,model
        return E_s
    else:
        E_s, propose_batch, s1length = calculate_sprime(spin_lattice, J2=0)
        del params,model
        return propose_batch, spin_lattice


def MCMC(Nsweep,spin_lattice,model,params,sign_rule_switch):
    ws = WS_CALCULATE(spin_lattice, model, params, sign_rule_switch)
    ele = calculate_local_energy(ws, spin_lattice, model,params, 0, sign_rule_switch)
    Es_list = []
    sweep_count = 0
    collected_samples = 0
    while collected_samples < Nsweep:
        sweep_count += 1
        propose_batch,spin_lattice = ele[0],ele[1]
        pickup_indice = np.random.randint(0, len(propose_batch))
        site_1, spin_2, site_2, spin_1 = propose_batch[pickup_indice]
        next_spin_lattice=np.copy(spin_lattice)
        next_spin_lattice[site_1[0],site_1[1]]=spin_2
        next_spin_lattice[site_2[0],site_2[1]]=spin_1
        ws_1 = WS_CALCULATE(next_spin_lattice, model, params, sign_rule_switch)
        P = (ws_1 / ws) ** 2
        r = np.random.uniform(0, 1)
        if r < P:
            del ele,ws
            ele = calculate_local_energy(ws_1, next_spin_lattice, model, params, 0, sign_rule_switch)
            spin_lattice = ele[1]
            ws = ws_1
        else:
            pass
        if sweep_count % 5 == 0:
            ele1 = calculate_local_energy(ws, spin_lattice, model, params, 1, sign_rule_switch)
            collected_samples += 1
            Es_list.append(ele1)
        if collected_samples%100==0:
            with open('progress','a') as f:
                f.write('{} samples collected\n'.format(collected_samples))
    return Es_list, spin_lattice