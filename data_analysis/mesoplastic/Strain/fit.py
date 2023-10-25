""" Every function that fit curve for stress-strain related data """

from typing import List, Tuple, Optional
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mesoplastic import compute

# Every plot will have the same presentation (fontsize, labelsize, ...)
plt.style.use('/home/victor/Documents/stage/data_analysis/presentation.mplstyle')

def flowFit(x: np.ndarray,
            A: float,
            n: float
            ) -> np.ndarray:
    """ Fit function """
    return  A * x**n

def YieldStressFit(x: np.ndarray,
                   sigma_y: float,
                   A: float,
                   n: float
                   ) -> np.ndarray:
    return sigma_y + A * x**n

def FlowCruveFit(list_gdot: np.ndarray,
                 sigma_xy_list: np.ndarray,
                 yield_stress: float,
                 color: str,
                 p0: List[int],
                 vmin: int = 0,
                 vmax: Optional[int] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """ Gets the optimal parameters for the flow curve """
    try:
        popt_flow, pcov_flow = curve_fit(flowFit, list_gdot[vmin:vmax], sigma_xy_list[vmin:vmax], p0)
    except RuntimeError:
        print("Couldn't find the optimal parameters")
        return np.zeros(2, dtype=float), np.zeros((2,2), dtype=float)
    corr = compute.getRsquared(list_gdot[vmin:vmax], sigma_xy_list[vmin:vmax], popt_flow, flowFit)
    plt.loglog(list_gdot[vmin:vmax], flowFit(list_gdot[vmin:vmax], *popt_flow), color=color, linestyle='--')
    print(corr, popt_flow)
    return popt_flow, pcov_flow