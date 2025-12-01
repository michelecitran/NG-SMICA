# NG-SMICA
I prepared this directory as a code sample for my application to the CMB group at the University of Oslo. 

It contains the multi-frequency multi-component binned bispectrum estimation explained in Citran et al. (arXiv:2511.22641).

The binned bispectrum for a single polarization is 

$$B^{d_1d_2d_3}_{i_1i_2i_3}$$

where the index $d$ stands for the $d$-th frequency channel while the index $i$ represents the $i$-th bin.

In the case of the presence of the two main foregrounds contaminants to CMB: dust and synchrotron, the bispectrum is parametrized as

$$B^{d_1d_2d_3}_{i_1i_2i_3}= A^{d_1, \text{CMB}}A^{d_2, \text{CMB}}A^{d_3, \text{CMB}}f_{\text{NL}}^{\text{th}} B^{\text{CMB}}_{i_1i_2i_3}$$
$$+A^{d_1, \text{dust}}A^{d_2, \text{dust}}A^{d_3, \text{dust}}B^{\text{dust}}_{i_1i_2i_3}$$
$$+A^{d_1, \text{sync}}A^{d_2, \text{sync}}A^{d_3, \text{sync}}B^{\text{sync}}_{i_1i_2i_3}$$

After estimating the mixing matrix of every component $A$ and the covariance matrix of the model $\textbf{C}$ we aim to estimate 

$$\theta = (f_{\text{NL}}^{\text{th}}, B^{\text{dust}}_{i_1i_2i_3}, B^{\text{sync}}_{i_1i_2i_3}) $$

Where $f_{\text{NL}}^{\text{th}}$ depends on the theoretical model chosen for the CMB bispectrum.

In binned_bisp_ext.py there is the code that I have written to compute the observed multi-frequency binned bispectrum needed to compute $\theta$.
