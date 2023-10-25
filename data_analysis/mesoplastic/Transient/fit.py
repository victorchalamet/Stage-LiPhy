""" Every function that fit curve for transient related data """

from typing import List, Tuple, Optional
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mesoplastic import compute

# Every plot will have the same presentation (fontsize, labelsize, ...)
plt.style.use('/home/victor/Documents/stage/data_analysis/presentation.mplstyle')

def powerFit(x: np.ndarray,
             a: float,
             b: float
             ) -> np.ndarray:
    """ Fit function """
    return a * x**b

def fractionFit(x: np.ndarray,
                a: float,
                b: float
                ) -> np.ndarray:
    """ Fit function """
    return (1/((a/x**-2)+(b/x**-4)))

def SigmaxyPowerFit(list_x: np.ndarray,
                    sigma_xy_list: np.ndarray,
                    color: str, 
                    p0: List = [1,1],
                    vmin: int = 0,
                    vmax: Optional[int] = None
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """ Gets the optimal parameters for a power fit """
    try:
        popt_power, pcov_power = curve_fit(powerFit, list_x[vmin:vmax], sigma_xy_list[vmin:vmax], p0)
    except RuntimeError:
        print("Couldn't find the optimal parameters")
        return np.zeros(2, dtype=int), np.zeros((2,2), dtype=int)
    corr = compute.getRsquared(list_x[vmin:vmax], sigma_xy_list[vmin:vmax], popt_power, powerFit)
    plt.loglog(list_x[vmin:vmax], powerFit(list_x[vmin:vmax], *popt_power), '*--',
               label=f'Slope={round(popt_power[1], 3)}, a={round(popt_power[0], 3)}, '+r'$r^{2}$'+f'={round(corr, 3)}',
               color=color)
    return popt_power, pcov_power

def SigmaxyFractionFit(list_x: np.ndarray,
                       sigma_xy_list: np.ndarray,
                       color: str,
                       ax,
                       p0: List = [1,1],
                       vmin: int = 0,
                       vmax: Optional[int] = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """ Gets the optimal parameters for a fraction fit """
    try:
        popt_fraction, pcov_fraction = curve_fit(fractionFit, list_x[vmin:vmax], sigma_xy_list[vmin:vmax], p0, bounds=([0,0],[np.inf,np.inf]))
    except RuntimeError:
        print("Couldn't find the optimal parameters")
        return np.zeros(2, dtype=int), np.zeros((2,2), dtype=int)
    # corr = abs(pcov_fraction[0][1]/np.sqrt(pcov_fraction[0][0]*pcov_fraction[1][1]))**2
    ax.loglog(list_x[vmin:vmax], fractionFit(list_x[vmin:vmax], *popt_fraction), '--', color=color)
    return popt_fraction, pcov_fraction



def SigmaxyTwoFit(list_x: np.ndarray,
                  sigma_xy_list: np.ndarray,
                  color: str,
                  indice_first: List[int],
                  indice_second: List[int],
                  p0: List[List[int]] = [[1,1],[1,1]],
                  vmax: Optional[int] = None
                  ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """ Gets the optimal parameters for a far/near field fit """
    try:
        popt_first, pcov_first = curve_fit(powerFit, list_x[indice_first[0]:indice_first[1]], sigma_xy_list[indice_first[0]:indice_first[1]], p0[0])
        popt_second, pcov_second = curve_fit(powerFit, list_x[indice_second[0]:indice_second[1]], sigma_xy_list[indice_second[0]:indice_second[1]], p0[1])
    except RuntimeError:
        print("Couldn't find the optimal parameters")
        return [np.zeros(2, dtype=int)], [np.zeros((2,2), dtype=int)]
    corr_first = abs(pcov_first[0][1]/np.sqrt(pcov_first[0][0]*pcov_first[1][1]))**2
    corr_second = abs(pcov_second[0][1]/np.sqrt(pcov_second[0][0]*pcov_second[1][1]))**2
    popt = [popt_first, popt_second]
    pcov = [pcov_first, pcov_second]

    plt.loglog(list_x[indice_first[0]:indice_first[1]],
               powerFit(list_x[indice_first[0]:indice_first[1]], *popt_first), '*--',
               label=f'a={round(popt[0][0], 3)}, slope={round(popt[0][1], 2)}, '+rf'$r^2={round(corr_first, 3)}$', color=color)
    plt.loglog(list_x[indice_second[0]:min(indice_second[1], vmax)], # type: ignore
               powerFit(list_x[indice_second[0]:min(indice_second[1], vmax)], *popt_second), '*--', # type: ignore
               label=f'b={round(popt[1][0], 7)}, slope={round(popt[1][1], 2)}, '+rf'$r^2={round(corr_second, 3)}$', color=color)
    return popt, pcov

def ScreeningFit(list_lambda: np.ndarray,
                 list_distance: np.ndarray,
                 p0: List[int],
                 vmin: int = 0,
                 vmax: Optional[int] = None):
    """ Plots the fit for the typical distance in terms of lambda """
    try:
        popt_screen, pcov_screen = curve_fit(powerFit, list_lambda[vmin:vmax], list_distance[vmin:vmax], p0)
    except RuntimeError:
        print("Couldn't find the optimal parameters")
