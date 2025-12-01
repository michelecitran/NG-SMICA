# NG-SMICA
I prepared this directory as a code sample for my application to the CMB group at the University of Oslo. As I work on a cluster (the code takes several hours to run) it is structured to work with parallel jobs with slurm.

It contains the multi-frequency multi-component binned bispectrum estimation explained in Citran et al. (arXiv:2511.22641).

The binned bispectrum for a single polarization is 

$$B^{d_1d_2d_3}_{i_1i_2i_3}$$

where the index $d$ stands for the $d$-th frequency channel while the index $i$ represents the $i$-th bin.

In the case of the presence of the two main polarization foreground contaminants to CMB: dust and synchrotron, the bispectrum is parametrized as

$$B^{d_1d_2d_3}_{i_1i_2i_3}= A^{d_1, \text{CMB}}A^{d_2, \text{CMB}}A^{d_3, \text{CMB}}f_{\text{NL}}^{\text{th}} B^{\text{CMB}}_{i_1i_2i_3}$$
$$+A^{d_1, \text{dust}}A^{d_2, \text{dust}}A^{d_3, \text{dust}}B^{\text{dust}}_{i_1i_2i_3}$$
$$+A^{d_1, \text{sync}}A^{d_2, \text{sync}}A^{d_3, \text{sync}}B^{\text{sync}}_{i_1i_2i_3}$$

After estimating with SMICA the mixing matrix of every component $A$ and the covariance matrix of the model $\textbf{C}$ we aim to estimate 

$$\theta = (f_{\text{NL}}^{\text{th}}, B^{\text{dust}}_{i_1i_2i_3}, B^{\text{sync}}_{i_1i_2i_3}) $$

Where $f_{\text{NL}}^{\text{th}}$ depends on the theoretical model chosen for the CMB bispectrum.

Assuming a gaussian likelihood describing the bispectrum in the weakly non-Gaussian case, it is possible to compute the parameters via the bispectrum variance 

$$ Var(B)^{d_1d_2d_3, d'_1d'1_2d'_3}_{l_1l_2l_3} \propto C^{d_1d'_1}_{l_1}C^{d_2d'_2}_{l_2}C^{d_3d'_3}_{l_3}$$

where the factor changes for the even and odd case.

1) In binned_bisp_ext.py there is the code that I wrote to compute the observed multi-frequency binned bispectrum needed to compute $\theta$.

Input: Frequency maps of the observed sky.

Output: Observed cross-frequency binned bispectrum.

Summary: It is based on the property of the triple (spin-weighted) spherical harmonics integral

$$\int d \Omega  \ _{s_1}\mathcal{Y}_{l_1m_1}(\Omega) \ _{s_2}\mathcal{Y}_{l_2m_2}(\Omega) \ _{s_3}\mathcal{Y}_{l_3m_3}(\Omega) \propto \begin{pmatrix} l_1 & l_2 & l_3 \\ 
m_1 & m_2 & m_3 \end{pmatrix}\begin{pmatrix} l_1 & l_2 & l_3 \\ 
-s_1 & -s_2 & -s_3 \end{pmatrix}$$

Which allow us to compute the bispectrum via the triple integral of filtered maps 

$$ _sM^d_{i} = \sum_{l\in\Delta}\sum_{m=-l}^l a^d_{lm} \ _s\mathcal{Y}_{lm}(\Omega)$$

2) In multi_ext.py there is the code that I wrote to compute the parameters given $A$ and $\textbf{C}$ from SMICA and the observed bispectrum from the previous code.

Input: $A$ and $\textbf{C}$ from SMICA, and the observed bispectrum $\hat{B}$.

Output: estimated $f_{\text{NL}}^{\text{th}}$ and $B^{\text{dust}}$ and  $B^{\text{sync}}$ both the even and odd sectors.

Summary: It is based on the likelihood

$$\mathcal{L}(B \ | \ \hat{C}, \hat{B}) \propto \exp\left( -1/2 (\hat{B}-B)^{d_1d_2d_3}_{i_1i_2i_3} (Var^{-1}(B))^{d_1d_2d_3, d'_1d'_2d'_3}_{i_1i_2i_3} (\hat{B}-B)^{d'_1d'_2d'_3}_{i_1i_2i_3} \right)$$

Hence there is a unique maximum likelihood solution.
