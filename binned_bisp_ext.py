import numpy as np
import healpy as hp
import sys
import opt_einsum as oe
from tqdm import tqdm

from MKparams import *

## Seed
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

## Functions

def get_bin_index(i1, i2, i3, nb):
   i = [i1, i2, i3]
   i.sort()
   index = i[0]*(i[0]**2-3*nb*i[0]+3*nb**2-1)/6 - i[1]*(i[1]-2*nb+1)/2 + i[2]
   # This is Sum[(nb-k+1)(nb-k+2)/2,{k,1,i[0]}] + Sum[nb-i[0]-k+1,{k,1,i[1]-i[0]}] + i[2]-i[1] 
   return int(index)

def alm_filter(alm_in, l_min_flt, l_max_flt):

   l_vec, m_vec = hp.Alm.getlm(hp.Alm.getlmax(alm_in.size))
   select_array = (l_vec >= l_min_flt) & (l_vec <= l_max_flt) 
   alm_out = np.zeros(hp.Alm.getsize(l_max_flt), dtype=alm_in.dtype)
   l_vec2, m_vec = hp.Alm.getlm(l_max_flt)
   select_array2 = (l_vec2 >= l_min_flt) & (l_vec2 <= l_max_flt)
   alm_out[select_array2] = alm_in[select_array]

   return alm_out

def create_filtered_map_list_s0(bin_list, alm_in, nside):
   
   filtered_map_list = []
   for l_interval in bin_list:
      l_left = l_interval[0]; l_right = l_interval[1]
      alm_filtered = alm_filter(alm_in, l_left, l_right)
      filtered_map = hp.alm2map(alm_filtered, nside)
      filtered_map_list.append(filtered_map)
   return filtered_map_list

def create_filtered_map_list_sm1(bin_list, alm_in, nside):
   
   filtered_map_list = []
   for l_interval in bin_list:
      l_left = l_interval[0]; l_right = l_interval[1]
      alm_filtered = alm_filter(alm_in, l_left, l_right)
      alms_filtered_0 = np.zeros(alm_filtered.shape)
      
      filtered_map_q,filtered_map_u = hp.alm2map_spin([alm_filtered,alms_filtered_0], nside=nside, spin = 1, lmax=l_right)
      filtered_map = filtered_map_q -1j * filtered_map_u
      
      filtered_map_list.append(filtered_map)
   return filtered_map_list

def create_filtered_map_list_s1(bin_list, alm_in, nside):
   
   filtered_map_list = []
   for l_interval in bin_list:
      l_left = l_interval[0]; l_right = l_interval[1]
      alm_filtered = alm_filter(alm_in, l_left, l_right)
      alms_filtered_0 = np.zeros(alm_filtered.shape)
      
      filtered_map_q,filtered_map_u = hp.alm2map_spin([-alm_filtered,alms_filtered_0], nside=nside, spin = 1, lmax=l_right)
      filtered_map = filtered_map_q +1j * filtered_map_u
      
      filtered_map_list.append(filtered_map)
   return filtered_map_list

def create_filtered_map_list_sm2(bin_list, alm_in, nside):
   
   filtered_map_list = []
   for l_interval in bin_list:
      l_left = l_interval[0]; l_right = l_interval[1]
      alm_filtered = alm_filter(alm_in, l_left, l_right)
      alms_filtered_0 = np.zeros(alm_filtered.shape)

      filtered_map_q,filtered_map_u = hp.alm2map_spin([-alm_filtered,alms_filtered_0], nside=nside, spin = 2, lmax=l_right)
      filtered_map = filtered_map_q -1j * filtered_map_u
      
      filtered_map_list.append(filtered_map)
   return filtered_map_list

def create_filtered_map_list_s2(bin_list, alm_in, nside):
   
   filtered_map_list = []
   for l_interval in bin_list:
      l_left = l_interval[0]; l_right = l_interval[1]
      alm_filtered = alm_filter(alm_in, l_left, l_right)
      alms_filtered_0 = np.zeros(alm_filtered.shape)

      filtered_map_q,filtered_map_u = hp.alm2map_spin([-alm_filtered,alms_filtered_0], nside=nside, spin = 2, lmax=l_right)
      filtered_map = filtered_map_q +1j * filtered_map_u
      
      filtered_map_list.append(filtered_map)
   return filtered_map_list

def IQU_2_TEB(maps, nside = 256, lmax = 767):
    """
    Convert IQU to IEB
    """
    alms = hp.map2alm(maps, lmax)
    #print(np.shape(alms))
    mapT = hp.alm2map(alms[0,:],nside)
    mapE = hp.alm2map(alms[1,:],nside)
    mapB = hp.alm2map(alms[2,:],nside)
    return np.array([mapT, mapE, mapB])


# Remember to check for a new run:

# select_freq: which frequencies to use from the LiteBIRD frequency list in freq_list

# bin_list : the binning choice 
# N.B. l_max = 100 cannot be changed unless new 3j symbols are re-computed

# pol : the polarization to use, at the moment only one at at time

# odd_flag : 1 if you want to include also the l1+l2+l3 odd part of the bispectrum, 0 if only even

# mask : Determine the mask, if unmasked sky do not multiply  

# bisp_name: Change directory based on content

# maps loading:

direc = "/sps/litebird/Users/mcitran/smica/"

odd_flag = 0

nside = 256 # consistent with LiteBIRD 
nside_filt = 128 # optimized for speed
npix_filt = hp.nside2npix(nside_filt)

freq_list = ["40GHz","50GHz","60GHz","68GHz","78GHz","89GHz","100GHz","119GHz","140GHz","166GHz","195GHz","235GHz","280GHz","337GHz","402GHz"]
select_freq = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
#select_freq = [0] # option for single component

freq_list = np.array(freq_list)[select_freq]
n_channel = len(freq_list)

map_cube_summed_freq = oe.contract_expression("ia,ja,ka->ijk", (n_channel,npix_filt,), (n_channel,npix_filt,), (n_channel,npix_filt,))

sensitivity = np.array([37.42, 33.46, 21.31, 16.87,12.07, 11.30, 10.34, 7.69,7.25, 5.57, 7.05, 10.79,13.80, 21.95, 47.45])
sensitivity = sensitivity[select_freq]
n_pix = hp.nside2npix(nside)
noise_amp = sensitivity*np.sqrt(n_pix/(4*np.pi))*np.pi/10800/np.sqrt(2)
noise_amp_pol = noise_amp*np.sqrt(2)

bin_list = [[2,5],[6,10],[11,15],[16,20],[21,30],[31,40],[41,50],[51,60],[61,70],[71,80],[81,90],[91,100]]
nbin = len(bin_list)

lmax = bin_list[-1][1]

pol_list = ["T","E","B"]
pol = 1 # T=0 E=1 B=2

print(seed)

direc_bisp = "/sps/litebird/Users/mcitran/smica/pysm_sims/bisp/CMB_DUST_SYNC_NOISE/"
bisp_name = direc_bisp + "Bisp_" + pol_list[pol] + "_" + str(nbin) + "nbin_" + str(lmax) + "lmax_" + "seed" + str(seed)

## Determine the mask
mask = np.load("/sps/litebird/Users/mcitran/smica/mask_60_C1.npy")
nside_mask = hp.get_nside(mask)
if nside_mask != nside:
   mask = hp.ud_grade(mask,nside, order_out='RING')

## Filtered maps

if odd_flag==1:
   print("odd part as well")
else:
   print("only even part")

binned_bispec_even = np.zeros( (int(nbin*(nbin+1)*(nbin+2)/6) , n_channel,n_channel,n_channel))
if odd_flag==1:
   binned_bispec_odd = np.zeros( (int(nbin*(nbin+1)*(nbin+2)/6) , n_channel,n_channel,n_channel),dtype=complex )

## d0 template
# dust_T = hp.fitsfunc.read_map(filename = direc + "pysm_sims/maps/templates/dust_t_new.fits")
# dust_T = hp.ud_grade(dust_T, nside)
# dust_Q = hp.fitsfunc.read_map(filename = direc + "pysm_sims/maps/templates/dust_q_new.fits")          
# dust_Q = hp.ud_grade(dust_Q, nside)
# dust_U = hp.fitsfunc.read_map(filename = direc + "pysm_sims/maps/templates/dust_u_new.fits")          
# dust_U = hp.ud_grade(dust_U, nside)

# dust_map_ref = np.array([dust_T, dust_Q, dust_U])
# dust_map_ref_TEB = IQU_2_TEB(dust_map_ref)

# dust_map_ref = 0
# dust_T=0;dust_Q=0;dust_U=0

## s0 template
# sync_T = hp.fitsfunc.read_map(filename = direc + "pysm_sims/maps/templates/synch_t_new.fits")
# sync_T = hp.ud_grade(sync_T, nside)
# sync_Q = hp.fitsfunc.read_map(filename = direc + "pysm_sims/maps/templates/synch_q_new.fits")          
# sync_Q = hp.ud_grade(sync_Q, nside)
# sync_U = hp.fitsfunc.read_map(filename = direc + "pysm_sims/maps/templates/synch_u_new.fits")          
# sync_U = hp.ud_grade(sync_U, nside)

# sync_map_ref = np.array([sync_T, sync_Q, sync_U])
# sync_map_ref_TEB = IQU_2_TEB(sync_map_ref)

# sync_map_ref = 0
# sync_T=0;sync_Q=0;sync_U=0


## Filtered computation
filt_map_list_s0 = []
if odd_flag==1:
   filt_map_list_sm1 = []
   filt_map_list_s1 = []
   filt_map_list_sm2 = []
   filt_map_list_s2 = []


#map_f = IQU_2_TEB(cmb_map)[pol] * mask
LB_freqs = np.array([40, 50, 60, 68, 78, 89, 100,                               #LB frequencies in GHz
                    119, 140, 166, 195, 235, 280, 337, 402])
nb_freqs = len(LB_freqs)                                                        #Number of frequency bands in LB

cmb_map = hp.read_map("/sps/litebird/Users/mcitran/smica/pysm_sims/maps/CMB/CMB_nside256_seed" + str(seed))

for i in range(n_channel):
   filt_map_list_s0.append([])
   if odd_flag==1:
      filt_map_list_s1.append([])
      filt_map_list_sm1.append([])
      filt_map_list_s2.append([])
      filt_map_list_sm2.append([])

   dust_map = hp.fitsfunc.read_map(filename = direc + "pysm_sims/maps/d0_maps/" + str(freq_list[i]) + "_nside256_DustSky", field = (0,1,2))
   sync_map = hp.fitsfunc.read_map(filename = direc + "pysm_sims/maps/s0_maps/" + str(freq_list[i]) + "_nside256_SyncSky", field = (0,1,2))
   noise = np.load(direc + "pysm_sims/maps/seeded_noise_maps/" + str(freq_list[i]) + "_nside256_noise_seed" + str(seed) + ".npy")
   
   map_f = cmb_map  + dust_map  + sync_map + [noise*noise_amp[i], noise*noise_amp_pol[i], noise*noise_amp_pol[i]] 
   
   #map_f = dust_map_ref_TEB[pol]*mask
   #map_f = sync_map_ref_TEB[pol]*mask
   
   map_f = IQU_2_TEB(map_f)[pol] * mask # masking directly in polarization
   

   alm = hp.map2alm(map_f)


   filt_map_list_s0[i] = create_filtered_map_list_s0(bin_list,alm,nside_filt)
   if odd_flag==1:
      filt_map_list_sm1[i] = create_filtered_map_list_sm1(bin_list,alm,nside_filt)
      filt_map_list_s1[i] = create_filtered_map_list_s1(bin_list,alm,nside_filt)
      filt_map_list_sm2[i] = create_filtered_map_list_sm2(bin_list,alm,nside_filt)
      filt_map_list_s2[i] = create_filtered_map_list_s2(bin_list,alm,nside_filt)

filt_map_list_s0 = np.array(filt_map_list_s0)
if odd_flag==1:
   filt_map_list_s1 = np.array(filt_map_list_s1, dtype=complex)
   filt_map_list_sm1 = np.array(filt_map_list_sm1, dtype=complex)
   filt_map_list_s2 = np.array(filt_map_list_s2, dtype=complex)
   filt_map_list_sm2 = np.array(filt_map_list_sm2, dtype=complex)



## Bispectrum computation

for i1 in range(nbin):
   for i2 in range(i1, nbin):
      for i3 in range(i2, nbin):
            if bin_list[i3][0] <= bin_list[i1][1] + bin_list[i2][1]:
               ii = get_bin_index(i1, i2, i3, nbin)

               binned_bispec_even[ii,:,:,:] = map_cube_summed_freq(filt_map_list_s0[:,i1],filt_map_list_s0[:,i2],filt_map_list_s0[:,i3])
               if odd_flag==1:
                  binned_bispec_odd[ii,:,:,:] = map_cube_summed_freq(filt_map_list_sm2[:,i1],filt_map_list_s1[:,i2],filt_map_list_s1[:,i3]) - map_cube_summed_freq(filt_map_list_s2[:,i1],filt_map_list_sm1[:,i2],filt_map_list_sm1[:,i3]) \
                                                + map_cube_summed_freq(filt_map_list_sm2[:,i3],filt_map_list_s1[:,i1],filt_map_list_s1[:,i2]) - map_cube_summed_freq(filt_map_list_s2[:,i3],filt_map_list_sm1[:,i1],filt_map_list_sm1[:,i2]) \
                                                + map_cube_summed_freq(filt_map_list_sm2[:,i2],filt_map_list_s1[:,i3],filt_map_list_s1[:,i1]) - map_cube_summed_freq(filt_map_list_s2[:,i2],filt_map_list_sm1[:,i3],filt_map_list_sm1[:,i1])
               
filt_map_list_s0 = []
filt_map_list_s1 = []
filt_map_list_sm1 = []
filt_map_list_s2 = []
filt_map_list_sm2 = []

binned_bispec_even = np.real(binned_bispec_even * 4. * np.pi / hp.nside2npix(nside_filt))
if odd_flag==1:
   binned_bispec_odd = np.imag(binned_bispec_odd * 4. * np.pi / hp.nside2npix(nside_filt) /6.0)

np.save(bisp_name + "_even", binned_bispec_even)
if odd_flag==1:
   np.save(bisp_name + "_odd", binned_bispec_odd)

print("\nNew template saved to file:", bisp_name)
