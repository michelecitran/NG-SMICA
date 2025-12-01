import numpy.linalg as nplin
import numpy as np
import sys
import opt_einsum as oe

## Seed
from MKparams import *
read_from_file = True
if read_from_file:
    params_filename = ""
    if len(sys.argv) > 1 and sys.argv[1][0] != "-":
      if sys.argv[1][-3:] == ".py":
         params_filename = sys.argv[1]
      else:
         params_filename = sys.argv[1] + ".py"
      i = params_filename.rfind("/")
      if i > -1:
         sys.path.append(params_filename[:i])
         exec("from " + params_filename[i+1:-3] + " import *")
      else:   
         exec("from " + params_filename[:-3] + " import *")

# Remember to check for a new run:

# Observed bispectrum:
# select_freq: which frequencies to use from the LiteBIRD frequency list in freq_list
# bin_list : the binning choice  
# pol : the polarization to use, at the moment only one at at time 
#  N.B. T is not supported for the moment! (more foregrounds present)
# bisp_name: Change directory based on content

# SMICA results: 
# direc_res: relative to observed bisp
# A mixing matrix dust,sync
# R covariance matrix model

# B_loc: the local shape binned bispectrum 

## Inputs

freq_list = ["40GHz","50GHz","60GHz","68GHz","78GHz","89GHz","100GHz","119GHz","140GHz","166GHz","195GHz","235GHz","280GHz","337GHz","402GHz"]
select_freq = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]

freq_list = np.array(freq_list)[select_freq]
n_channel = len(freq_list)


bin_list = [[2,5],[6,10],[11,15],[16,20],[21,30],[31,40],[41,50],[51,60],[61,70],[71,80],[81,90],[91,100]]
nbin = len(bin_list)

lmax = bin_list[-1][1]

pol_list = ["T","E","B"]
pol = 1 # E=1 B=2

direc_bisp = "/sps/litebird/Users/mcitran/smica/pysm_sims/bisp/CMB_DUST_SYNC_NOISE/"

bisp_name = direc_bisp + "Bisp_" + pol_list[pol] + "_" + str(nbin) + "nbin_" + str(lmax) + "lmax_" + "seed" + str(seed)

B_obs_even = np.load(bisp_name + "_even.npy")
B_obs_odd = np.load(bisp_name + "_odd.npy")

B_obs = [B_obs_even, B_obs_odd]

direc_res = "/sps/litebird/Users/mcitran/smica/smica_res/ng/3c/15f_corr_bisp/"

A_d =  np.load(direc_res + "A_d_seed" + str(seed) + ".npy")
A_s =  np.load(direc_res + "A_s_seed" + str(seed) + ".npy")

A = [A_d, A_s]

R_tot =  np.load(direc_res + "R_seed" + str(seed) + ".npy")
if pol == 1:
    R_m = R_tot[:,:,99:]
elif pol == 2:
    R_m = R_tot[:,:,:99]

B_loc = np.load("/sps/litebird/Users/mcitran/smica/new_sims/bisp_local_E.npy")

##  Tensor contractions
var_contr = oe.contract_expression("ai,bj,ck->abcijk", (n_channel,n_channel), (n_channel,n_channel), (n_channel,n_channel))
T_contr = oe.contract_expression("a,b,c,abcijk,i,j,k", (n_channel,),(n_channel,),(n_channel,), (n_channel,n_channel,n_channel, n_channel,n_channel,n_channel), (n_channel,),(n_channel,),(n_channel,))
T_hat_contr = oe.contract_expression("a,b,c,abcijk,ijk", (n_channel,),(n_channel,),(n_channel,), (n_channel,n_channel,n_channel, n_channel,n_channel,n_channel), (n_channel,n_channel,n_channel))

T_hat_noise_contr = oe.contract_expression("da,db,dc,abcijk,ijk->d", (n_channel,n_channel),(n_channel,n_channel),(n_channel,n_channel), (n_channel,n_channel,n_channel, n_channel,n_channel,n_channel), (n_channel,n_channel,n_channel))
T_comp_noise_contr = oe.contract_expression("da,db,dc,abcijk,i,j,k->d", (n_channel,n_channel),(n_channel,n_channel),(n_channel,n_channel), (n_channel,n_channel,n_channel, n_channel,n_channel,n_channel), (n_channel,), (n_channel,), (n_channel,))
T_noise_noise_contr = oe.contract_expression("da,db,dc,abcijk,ip,jp,kp->dp", (n_channel,n_channel),(n_channel,n_channel),(n_channel,n_channel), (n_channel,n_channel,n_channel, n_channel,n_channel,n_channel), (n_channel,n_channel), (n_channel,n_channel), (n_channel,n_channel))

## Functions

# precomputed 3j symbols
h_even = np.loadtxt("h_tilde_000.txt")
def get_h_000(l1, l2, l3):
    mask = (h_even[:, 0] == l1) & (h_even[:, 1] == l2) & (h_even[:, 2] == l3)
    result = h_even[mask][:, 3]
    if result.size > 0:
        return (result[0])**2  
    else: 
        raise ValueError("Invalid even triplet", l1,l2,l3)

h_odd = np.loadtxt("h_tilde_2m1m1.txt")
def get_h_2m1m1(l1, l2, l3):
    mask = (h_odd[:, 0] == l1) & (h_odd[:, 1] == l2) & (h_odd[:, 2] == l3)
    result = h_odd[mask][:, 3]
    if result.size > 0:
        return (result[0])**2  
    else: 
        raise ValueError("Invalid odd triplet", l1,l2,l3)

def get_bin_index(i1, i2, i3, nb):
   i = [i1, i2, i3]
   i.sort()
   index = i[0]*(i[0]**2-3*nb*i[0]+3*nb**2-1)/6 - i[1]*(i[1]-2*nb+1)/2 + i[2]
   # This is Sum[(nb-k+1)(nb-k+2)/2,{k,1,i[0]}] + Sum[nb-i[0]-k+1,{k,1,i[1]-i[0]}] + i[2]-i[1] 
   return int(index)

def find_dep(condition):
    res, = np.nonzero(np.ravel(condition))
    return res

# stable pseudo inversion as in SMICA
def invert_triplet(Var_ii):
    
    n_channel = Var_ii.shape[0]

    V_ii = Var_ii.reshape(n_channel**3, n_channel**3)

    V_ii = np.mat(V_ii)
    V_ii = 0.5 * (V_ii + V_ii.T)

    fudge_factor = 1000
    u,v = nplin.eigh(V_ii) 
    eps = 2e-16
    pos = find_dep(u > fudge_factor*eps*sum(u) )

    if np.size(pos) == 0:
            print("numerical issue in kullback")
            vp = v
            up = u
    else:
            vp = v[:,pos]
            up = u[pos]

    iV_ii = np.array(vp @ np.diag(1.0 / up) @ vp.T)

    return iV_ii.reshape(n_channel,n_channel,n_channel, n_channel,n_channel,n_channel)

def comp_T_and_T_hat_E(A,R,B_obs,bin_list,n_channel):
    
    A_cmb = np.ones(n_channel)
    A_d = A[0]
    A_s = A[1]

    
    B_obs_even = B_obs[0]
    B_obs_odd = B_obs[1]

    nbin = len(bin_list)
    ntriplets = get_bin_index(nbin,nbin,nbin,nbin)    

    T_hat_cmb_even = np.zeros((ntriplets)) # CMB is E pol even only
    T_hat_d_even = np.zeros((ntriplets)) 
    T_hat_s_even = np.zeros((ntriplets)) 
    T_hat_d_odd = np.zeros((ntriplets)) 
    T_hat_s_odd = np.zeros((ntriplets)) 

    T_cmbcmb_even = np.zeros((ntriplets))
    T_cmbd_even = np.zeros((ntriplets))
    T_cmbs_even = np.zeros((ntriplets))
    T_dd_even = np.zeros((ntriplets))
    T_ss_even = np.zeros((ntriplets))
    T_sd_even = np.zeros((ntriplets))

    T_dd_odd = np.zeros((ntriplets))
    T_ss_odd = np.zeros((ntriplets))
    T_sd_odd = np.zeros((ntriplets))


    for i1 in range(nbin):
        l1_l = bin_list[i1][0]
        l1_r = bin_list[i1][1]+1
        for i2 in range(i1,nbin):
            l2_l = bin_list[i2][0]
            l2_r = bin_list[i2][1]+1
            for i3 in range(i2,nbin):
                l3_l = bin_list[i3][0]
                l3_r = bin_list[i3][1]+1

                if bin_list[i3][0] <= bin_list[i1][1] + bin_list[i2][1]:

                    Var_odd_ii = np.zeros((n_channel,n_channel,n_channel, n_channel,n_channel,n_channel))
                    Var_even_ii = np.zeros((n_channel,n_channel,n_channel, n_channel,n_channel,n_channel))

                    ii = get_bin_index(i1,i2,i3,nbin)

                    B_obs_even_ii = B_obs_even[ii,:,:,:]
                    B_obs_odd_ii = B_obs_odd[ii,:,:,:]

                    if i1 == i2 and i2== i3:
                        factor = 6
                    elif i1==i2 or i2==i3 or i1==i3:
                        factor = 2
                    else:
                        factor = 1

                    for l1 in range(l1_l,l1_r):
                        for l2 in range(l2_l,l2_r):
                            for l3 in range(l3_l,l3_r):

                                if np.abs(l1-l2)<=l3 and l3<=l1+l2:
                                    if (l1+l2+l3)%2==0:
                                        Var_even_ii += (2*l1+1)*(2*l2+1)*(2*l3+1)/4/np.pi * get_h_000(l1,l2,l3) * var_contr(R[:,:,l1-2],R[:,:,l2-2],R[:,:,l3-2]) * factor
                                    elif l1!=l2 and l2!=l3 and l1!=l3:
                                        Var_odd_ii += (2*l1+1)*(2*l2+1)*(2*l3+1)/4/np.pi * get_h_2m1m1(l1,l2,l3) * var_contr(R[:,:,l1-2],R[:,:,l2-2],R[:,:,l3-2]) * factor/9.0
                
                    iV_ii_even = invert_triplet(Var_even_ii) 
                    iV_ii_odd = invert_triplet(Var_odd_ii)

                    T_hat_cmb_even[ii] = T_hat_contr(A_cmb,A_cmb,A_cmb, iV_ii_even , B_obs_even_ii)
                    T_hat_d_even[ii] = T_hat_contr(A_d,A_d,A_d, iV_ii_even , B_obs_even_ii)
                    T_hat_s_even[ii] = T_hat_contr(A_s,A_s,A_s, iV_ii_even, B_obs_even_ii)

                    T_hat_d_odd[ii] = T_hat_contr(A_d,A_d,A_d, iV_ii_odd , B_obs_odd_ii)
                    T_hat_s_odd[ii] = T_hat_contr(A_s,A_s,A_s, iV_ii_odd, B_obs_odd_ii)

                    T_cmbcmb_even[ii] = T_contr(A_cmb,A_cmb,A_cmb, iV_ii_even, A_cmb,A_cmb,A_cmb)
                    T_cmbd_even[ii] = T_contr(A_cmb,A_cmb,A_cmb, iV_ii_even, A_d,A_d,A_d)
                    T_cmbs_even[ii] = T_contr(A_cmb,A_cmb,A_cmb, iV_ii_even, A_s,A_s,A_s)
                    T_dd_even[ii] = T_contr(A_d,A_d,A_d, iV_ii_even, A_d,A_d,A_d)
                    T_ss_even[ii] = T_contr(A_s,A_s,A_s, iV_ii_even, A_s,A_s,A_s)
                    T_sd_even[ii] = T_contr(A_s,A_s,A_s, iV_ii_even, A_d,A_d,A_d)

                    T_dd_odd[ii] = T_contr(A_d,A_d,A_d, iV_ii_odd, A_d,A_d,A_d)
                    T_ss_odd[ii] = T_contr(A_s,A_s,A_s, iV_ii_odd, A_s,A_s,A_s)
                    T_sd_odd[ii] = T_contr(A_s,A_s,A_s, iV_ii_odd, A_d,A_d,A_d)

    T_hat_e =[T_hat_cmb_even, T_hat_d_even, T_hat_s_even]
    T_hat_o =[T_hat_d_odd, T_hat_s_odd]

    T_e = [T_cmbcmb_even, T_cmbd_even, T_cmbs_even, T_dd_even, T_ss_even, T_sd_even]
    T_o = [T_dd_odd, T_ss_odd, T_sd_odd]

    return  T_e, T_o, T_hat_e, T_hat_o

def comp_T_and_T_hat_B(A,R,B_obs,bin_list,n_channel):
    
    A_d = A[0]
    A_s = A[1]

    
    B_obs_even = B_obs[0]
    B_obs_odd = B_obs[1]

    nbin = len(bin_list)
    ntriplets = get_bin_index(nbin,nbin,nbin,nbin)    

    T_hat_d_even = np.zeros((ntriplets)) 
    T_hat_s_even = np.zeros((ntriplets)) 
    T_hat_d_odd = np.zeros((ntriplets)) 
    T_hat_s_odd = np.zeros((ntriplets)) 

    T_dd_even = np.zeros((ntriplets))
    T_ss_even = np.zeros((ntriplets))
    T_sd_even = np.zeros((ntriplets))

    T_dd_odd = np.zeros((ntriplets))
    T_ss_odd = np.zeros((ntriplets))
    T_sd_odd = np.zeros((ntriplets))


    for i1 in range(nbin):
        l1_l = bin_list[i1][0]
        l1_r = bin_list[i1][1]+1
        for i2 in range(i1,nbin):
            l2_l = bin_list[i2][0]
            l2_r = bin_list[i2][1]+1
            for i3 in range(i2,nbin):
                l3_l = bin_list[i3][0]
                l3_r = bin_list[i3][1]+1

                if bin_list[i3][0] <= bin_list[i1][1] + bin_list[i2][1]:

                    Var_odd_ii = np.zeros((n_channel,n_channel,n_channel, n_channel,n_channel,n_channel))
                    Var_even_ii = np.zeros((n_channel,n_channel,n_channel, n_channel,n_channel,n_channel))

                    ii = get_bin_index(i1,i2,i3,nbin)

                    B_obs_even_ii = B_obs_even[ii,:,:,:]
                    B_obs_odd_ii = B_obs_odd[ii,:,:,:]

                    if i1 == i2 and i2== i3:
                        factor = 6
                    elif i1==i2 or i2==i3 or i1==i3:
                        factor = 2
                    else:
                        factor = 1

                    for l1 in range(l1_l,l1_r):
                        for l2 in range(l2_l,l2_r):
                            for l3 in range(l3_l,l3_r):

                                if np.abs(l1-l2)<=l3 and l3<=l1+l2:
                                    if (l1+l2+l3)%2==0:
                                        Var_even_ii += (2*l1+1)*(2*l2+1)*(2*l3+1)/4/np.pi * get_h_000(l1,l2,l3) * var_contr(R[:,:,l1-2],R[:,:,l2-2],R[:,:,l3-2]) * factor
                                    elif l1!=l2 and l2!=l3 and l1!=l3:
                                        Var_odd_ii += (2*l1+1)*(2*l2+1)*(2*l3+1)/4/np.pi * get_h_2m1m1(l1,l2,l3) * var_contr(R[:,:,l1-2],R[:,:,l2-2],R[:,:,l3-2]) * factor/9.0
                
                    iV_ii_even = invert_triplet(Var_even_ii) 
                    iV_ii_odd = invert_triplet(Var_odd_ii)

                    T_hat_d_even[ii] = T_hat_contr(A_d,A_d,A_d, iV_ii_even , B_obs_even_ii)
                    T_hat_s_even[ii] = T_hat_contr(A_s,A_s,A_s, iV_ii_even, B_obs_even_ii)

                    T_hat_d_odd[ii] = T_hat_contr(A_d,A_d,A_d, iV_ii_odd , B_obs_odd_ii)
                    T_hat_s_odd[ii] = T_hat_contr(A_s,A_s,A_s, iV_ii_odd, B_obs_odd_ii)

                    T_dd_even[ii] = T_contr(A_d,A_d,A_d, iV_ii_even, A_d,A_d,A_d)
                    T_ss_even[ii] = T_contr(A_s,A_s,A_s, iV_ii_even, A_s,A_s,A_s)
                    T_sd_even[ii] = T_contr(A_s,A_s,A_s, iV_ii_even, A_d,A_d,A_d)

                    T_dd_odd[ii] = T_contr(A_d,A_d,A_d, iV_ii_odd, A_d,A_d,A_d)
                    T_ss_odd[ii] = T_contr(A_s,A_s,A_s, iV_ii_odd, A_s,A_s,A_s)
                    T_sd_odd[ii] = T_contr(A_s,A_s,A_s, iV_ii_odd, A_d,A_d,A_d)

    T_hat_e =[T_hat_d_even, T_hat_s_even]
    T_hat_o =[T_hat_d_odd, T_hat_s_odd]

    T_e = [T_dd_even, T_ss_even, T_sd_even]
    T_o = [T_dd_odd, T_ss_odd, T_sd_odd]

    return  T_e, T_o, T_hat_e, T_hat_o

print("Starting multi-frequency bispectrum estimation...")

## Tensor estimation 

if pol == 1: 
    T_e, T_o, T_hat_e, T_hat_o = comp_T_and_T_hat_E(A,R_m,B_obs,bin_list,n_channel)
    
    T_hat_cmb_even, T_hat_d_even, T_hat_s_even = T_hat_e
    T_hat_d_odd, T_hat_s_odd = T_hat_o
    T_cmbcmb_even, T_cmbd_even, T_cmbs_even, T_dd_even, T_ss_even, T_sd_even = T_e
    T_dd_odd, T_ss_odd, T_sd_odd = T_o

elif pol == 2:
    T_e, T_o, T_hat_e, T_hat_o = comp_T_and_T_hat_B(A,R_m,B_obs,bin_list,n_channel)

    T_hat_d_even, T_hat_s_even = T_hat_e
    T_hat_d_odd, T_hat_s_odd = T_hat_o
    T_dd_even, T_ss_even, T_sd_even = T_e
    T_dd_odd, T_ss_odd, T_sd_odd = T_o

np.save(direc_res + "T_e_" + pol_list[pol]  + "_seed" + str(seed), T_e)
np.save(direc_res + "T_o_" + pol_list[pol]  + "_seed" + str(seed), T_o)
np.save(direc_res + "T_hat_e_" + pol_list[pol]  + "_seed" + str(seed), T_hat_e)
np.save(direc_res + "T_hat_o_" + pol_list[pol]  + "_seed" + str(seed), T_hat_o)


## multidetector bisp ext
ntriplets = get_bin_index(nbin,nbin,nbin,nbin)  
B_d_even = np.zeros((ntriplets))
B_d_odd = np.zeros((ntriplets))

B_s_even = np.zeros((ntriplets))
B_s_odd = np.zeros((ntriplets))


if pol ==1:

    num = 0.0
    denum = 0.0

    for i in range(ntriplets):

        # Condition for existence and uniqueness of the solution
        if T_cmbcmb_even[i]>0 and T_dd_even[i]>0 and T_ss_even[i]>0 and (T_ss_even[i]*T_dd_even[i]-T_sd_even[i]*T_sd_even[i])>0 :

            num += B_loc[i]*T_hat_cmb_even[i]    \
                  -B_loc[i]*T_cmbd_even[i] *(T_hat_d_even[i]*T_ss_even[i]-T_sd_even[i]*T_hat_s_even[i])/(T_dd_even[i]*T_ss_even[i]-T_sd_even[i]*T_sd_even[i])   \
                  -B_loc[i]*T_cmbs_even[i] *(T_hat_s_even[i]*T_dd_even[i]-T_sd_even[i]*T_hat_d_even[i])/(T_dd_even[i]*T_ss_even[i]-T_sd_even[i]*T_sd_even[i])   \
                
            denum += B_loc[i]*B_loc[i]*T_cmbcmb_even[i]    \
                   -B_loc[i]*B_loc[i]*T_cmbd_even[i] *(T_cmbd_even[i]*T_ss_even[i]-T_sd_even[i]*T_cmbs_even[i])/(T_dd_even[i]*T_ss_even[i]-T_sd_even[i]*T_sd_even[i])   \
                   -B_loc[i]*B_loc[i]*T_cmbs_even[i] *(T_cmbs_even[i]*T_dd_even[i]-T_sd_even[i]*T_cmbd_even[i])/(T_dd_even[i]*T_ss_even[i]-T_sd_even[i]*T_sd_even[i])   \
                
    f_NL =   num/denum

    for i in range(ntriplets):

        if T_dd_even[i]!=0 : # Non-excluded triplets

            if T_cmbcmb_even[i]>0 and T_dd_even[i]>0 and T_ss_even[i]>0 and (T_ss_even[i]*T_dd_even[i]-T_sd_even[i]*T_sd_even[i])>0 :        

                B_d_even[i] = (T_hat_d_even[i]*T_ss_even[i]-T_sd_even[i]*T_hat_s_even[i])/(T_dd_even[i]*T_ss_even[i]-T_sd_even[i]*T_sd_even[i]) \
                                -f_NL*B_loc[i]*(T_cmbd_even[i]*T_ss_even[i]-T_sd_even[i]*T_cmbs_even[i])/(T_dd_even[i]*T_ss_even[i]-T_sd_even[i]*T_sd_even[i])

                B_s_even[i] = (T_hat_s_even[i]*T_dd_even[i]-T_sd_even[i]*T_hat_d_even[i])/(T_dd_even[i]*T_ss_even[i]-T_sd_even[i]*T_sd_even[i]) \
                                -f_NL*B_loc[i]*(T_cmbs_even[i]*T_dd_even[i]-T_sd_even[i]*T_cmbd_even[i])/(T_dd_even[i]*T_ss_even[i]-T_sd_even[i]*T_sd_even[i])      

            else:
                raise ValueError("Error (even) it should be a valid triplet:", i, T_cmbcmb_even[i]>0 , T_dd_even[i]>0 , T_ss_even[i]>0 , (T_ss_even[i]*T_dd_even[i]-T_sd_even[i]*T_sd_even[i])>0)


            if T_dd_odd[i]>0 and (T_ss_odd[i]*T_dd_even[i]-T_sd_odd[i]*T_sd_odd[i])>0 :        

                B_d_odd[i] = (T_hat_d_odd[i]*T_ss_odd[i]-T_sd_odd[i]*T_hat_s_odd[i])/(T_dd_odd[i]*T_ss_odd[i]-T_sd_odd[i]*T_sd_odd[i])

                B_s_odd[i] = (T_hat_s_odd[i]*T_dd_odd[i]-T_sd_odd[i]*T_hat_d_odd[i])/(T_dd_odd[i]*T_ss_odd[i]-T_sd_odd[i]*T_sd_odd[i])                

            else:
                raise ValueError("Error (odd) it should be a valid triplet:", i, T_dd_odd[i]>0 , T_ss_odd[i]>0 , (T_ss_odd[i]*T_dd_odd[i]-T_sd_odd[i]*T_sd_odd[i])>0)
            
elif pol==2:
     
     for i in range(ntriplets):

        if T_dd_even[i]!=0 : # Non-excluded triplets
                # Condition for existence and uniqueness of the solution
            if T_dd_even[i]>0 and T_ss_even[i]>0 and (T_ss_even[i]*T_dd_even[i]-T_sd_even[i]*T_sd_even[i])>0 :        

                B_d_even[i] = (T_hat_d_even[i]*T_ss_even[i]-T_sd_even[i]*T_hat_s_even[i])/(T_dd_even[i]*T_ss_even[i]-T_sd_even[i]*T_sd_even[i]) 

                B_s_even[i] = (T_hat_s_even[i]*T_dd_even[i]-T_sd_even[i]*T_hat_d_even[i])/(T_dd_even[i]*T_ss_even[i]-T_sd_even[i]*T_sd_even[i])    

            else:
                raise ValueError("Error (even) it should be a valid triplet:", i, T_dd_even[i]>0 , T_ss_even[i]>0 , (T_ss_even[i]*T_dd_even[i]-T_sd_even[i]*T_sd_even[i])>0)

            if T_dd_odd[i]>0 and (T_ss_odd[i]*T_dd_even[i]-T_sd_odd[i]*T_sd_odd[i])>0 :        

                B_d_odd[i] = (T_hat_d_odd[i]*T_ss_odd[i]-T_sd_odd[i]*T_hat_s_odd[i])/(T_dd_odd[i]*T_ss_odd[i]-T_sd_odd[i]*T_sd_odd[i])

                B_s_odd[i] = (T_hat_s_odd[i]*T_dd_odd[i]-T_sd_odd[i]*T_hat_d_odd[i])/(T_dd_odd[i]*T_ss_odd[i]-T_sd_odd[i]*T_sd_odd[i])                

            else:
                raise ValueError("Error (odd) it should be a valid triplet:", i, T_dd_odd[i]>0 , T_ss_odd[i]>0 , (T_ss_odd[i]*T_dd_odd[i]-T_sd_odd[i]*T_sd_odd[i])>0)
            
if pol==1:
    np.save(direc_res + "f_NL_" + pol_list[pol]  + "_loc" + str(seed), f_NL)

np.save(direc_res + "B_d_" + pol_list[pol]  + "_even_seed" + str(seed), B_d_even)
np.save(direc_res + "B_d_" + pol_list[pol]  + "_odd_seed" + str(seed), B_d_odd)

np.save(direc_res + "B_s_" + pol_list[pol]  + "_even_seed" + str(seed), B_s_even)
np.save(direc_res + "B_s_" + pol_list[pol]  + "_odd_seed" + str(seed), B_s_odd)


