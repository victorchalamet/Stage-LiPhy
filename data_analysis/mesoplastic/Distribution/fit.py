""" Every function that fit curve for activity distribution related data """

from typing import List, Tuple, Optional, Union
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mesoplastic import compute

# Every plot will have the same presentation (fontsize, labelsize, ...)
plt.style.use('/home/victor/Documents/stage/data_analysis/presentation.mplstyle')

def expFit(x: np.ndarray,
           a: float,
           l: float
           ) -> np.ndarray:
    return a * np.exp(-x/l)

def SpatialAutoCorrelationFit(list_x: np.ndarray,
                              all_acorr_h_acc: Union[np.ndarray, List],
                              color: str,
                              p0: List[int] = [1,1],
                              vmin: int = 0,
                              vmax: Optional[int] = None
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """ Gets the optimal parameters of the autocorrelation curve for an exponential fit `a * b**x` """
    try:
        popt_corr, pcov_corr = curve_fit(expFit, list_x[vmin:vmax], all_acorr_h_acc[vmin:vmax], p0)
    except RuntimeError:
        print("Couldn't find the optimal parameters")
        return np.zeros(2, dtype=int), np.zeros((2,2), dtype=int)
    corr = compute.getRsquared(list_x[vmin:vmax], all_acorr_h_acc[vmin:vmax], popt_corr, expFit)
    if corr < 0.92:
        vmax = 4
    plt.semilogy(list_x[vmin:vmax], expFit(list_x[vmin:vmax], *popt_corr), '--', color=color)
    print(corr, popt_corr)
    return popt_corr, pcov_corr