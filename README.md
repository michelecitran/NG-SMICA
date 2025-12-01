# NG-SMICA
I prepared this directory as a code sample for my application to the CMB group at the University of Oslo. As I work on a cluster (the code takes several hours to run) it is structured to work with parallel jobs with slurm.

It contains the multi-frequency multi-component binned bispectrum estimation explained in Citran et al. (arXiv:2511.22641).

The binned bispectrum for a single polarization is 

$$B^{d_1d_2d_3}_{i_1i_2i_3}$$

where the index $d$ stands for the $d$-th frequency channel while the index $i$ represents the $i$-th bin.

In the case of the presence of the two main foregrounds contaminants to CMB: dust and synchrotron, the bispectrum is parametrized as

$$B^{d_1d_2d_3}_{i_1i_2i_3}= A^{d_1, \text{CMB}}A^{d_2, \text{CMB}}A^{d_3, \text{CMB}}f_{\text{NL}}^{\text{th}} B^{\text{CMB}}_{i_1i_2i_3}$$
$$+A^{d_1, \text{dust}}A^{d_2, \text{dust}}A^{d_3, \text{dust}}B^{\text{dust}}_{i_1i_2i_3}$$
$$+A^{d_1, \text{sync}}A^{d_2, \text{sync}}A^{d_3, \text{sync}}B^{\text{sync}}_{i_1i_2i_3}$$

After estimating with SMICA the mixing matrix of every component $A$ and the covariance matrix of the model $\textbf{C}$ we aim to estimate 

$$\theta = (f_{\text{NL}}^{\text{th}}, B^{\text{dust}}_{i_1i_2i_3}, B^{\text{sync}}_{i_1i_2i_3}) $$

Where $f_{\text{NL}}^{\text{th}}$ depends on the theoretical model chosen for the CMB bispectrum.

Assuming a gaussian likelihood describing the bispectrum in the weakly non-Gaussian case, it is possible to compute the parameters via the bispectrum variance

$$ Var(B)^{d_1d_2d_3, d'_1d'1_2d'_3}_{l_1l_2l_3} \propto C^{d_1d'_1}_{l_1}C^{d_2d'_2}_{l_2}C^{d_3d'_3}_{l_3}$$

where the proportional factor changes for the even and odd case.

In binned_bisp_ext.py there is the code that I wrote to compute the observed multi-frequency binned bispectrum needed to compute $\theta$.

Input: Frequency maps of the observed sky.

Output: Observed cross-frequency bispectrum.

In multi_ext.py there is the code that I wrote to compute the parameters given $A$ and $\textbf{C}$ from SMICA and the observed bispectrum from the previous code.

Input: $A$ and $\textbf{C}$ from SMICA, and the observed bispectrum from the other code.

Output: estimated $f_{\text{NL}}^{\text{th}}$ and $B^{\text{dust}}_{i_1i_2i_3}$ and  $B^{\text{sync}}_{i_1i_2i_3})$ both even and odd.
