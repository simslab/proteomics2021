
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.sandbox.stats.multicomp import multipletests

# function for proteomics differential expression using Welch's t-test
# g1 and g2 are the normalized intensity matrices for the two groups being compared
# fact is the coefficient applied to the normalized values before adding a pseudo-count

# The function returns a tuple. The first element is an array of log2-fold-change values, the 
# second is an array of p-values, the third is an array of FDR-corrected p-values, and the fourth
# is an array of masks where False indicates that a gene was undetected in both groups.

def ttest_diffex_linlog_wzero(g1,g2,fact):
    lfcs=[]
    pvs=[]
    mask=[]
    for v1,v2 in zip(g1,g2):
        t1=sum(v1)
        t2=sum(v2)
        if t1+t2>0.0:
            stat,pv=ttest_ind(np.log2(v1*fact+1.),np.log2(v2*fact+1.),equal_var=False) # Welch's t-teset
            if t1==0.:
                lfc=np.Inf
            elif t2==0.:
                lfc=-np.Inf
            else:
                lfc = np.log2(np.mean(v2)/np.mean(v1))
            pvs.append(pv)
            mask.append(True)
            lfcs.append(lfc)
        else:
            mask.append(False)
    qvs = multipletests(pvs,method='fdr_bh')[1]
    return np.array(lfcs),np.array(pvs),qvs,np.array(mask)
