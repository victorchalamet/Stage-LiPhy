from typing import Callable, Union, List, Tuple
import numpy as np
import math
import os
from mesoplastic import utils

def SpatialAutoCorrelation(h_acc: np.ndarray
                           ) -> np.ndarray:
    """ Computes the spatial auto-correlation of the activity using fft """
    mean = np.mean(h_acc)
    var = np.var(h_acc)
    normalized_h_acc = h_acc - mean
    fft_h_acc = np.fft.fft2(normalized_h_acc)
    spectrum_h_acc = np.abs(fft_h_acc) ** 2
    acorr_h_acc = (np.fft.ifft2(spectrum_h_acc).real / var) / len(normalized_h_acc)
    return acorr_h_acc

def getRsquared(xdata: Union[np.ndarray, List[int]],
                ydata: Union[np.ndarray, List],
                popt: np.ndarray, 
                f: Callable
                ) -> float:
    """ Compute the value of r squared """
    residuals = ydata - f(xdata, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ydata-np.mean(ydata))**2)
    return 1 - (ss_res / ss_tot)

def ComputeSDDistance(list_popt: List[np.ndarray],
                      list_pcov: List[np.ndarray],
                      option: str
                      ) -> np.ndarray:
    """ Compute the standard deviation of the typical distance of screening """
    list_SD_distance = np.array([])
    for i, element in enumerate(list_pcov):
        if option=='fraction':
            a = list_popt[i][0]
            b = list_popt[i][1]
            delta_a = math.sqrt(element[0][0])
            delta_b = math.sqrt(element[1][1])
            delta_x = math.sqrt(b/a)*(b*delta_a - a*delta_b)/(2*b**2)
        if option=='two':
            a = list_popt[i][0][0]
            b = list_popt[i][1][0]
            delta_a = math.sqrt(element[0][0][0])
            delta_b = math.sqrt(element[1][0][0])
            delta_x = math.sqrt(a/b)*(a*delta_b - b*delta_a)/(2*a**2)
        list_SD_distance = utils.append(list_SD_distance, abs(delta_x), 0)
    return list_SD_distance

def DeltaSigma(list_sigma_xy: np.ndarray
               ) -> List[List[float]]:
    """ Computes the value of each stress drops """
    list_delta_sigma: List[List[float]] = []
    file_created = False
    if not os.path.exists('/home/victor/Documents/stage/ComputedData/DeltaSigmaStressDrops.dat'):
        file = open('/home/victor/Documents/stage/ComputedData/DeltaSigmaStressDrops.dat', 'w')
        file_created = True
    for sigma_xy in list_sigma_xy:
        derivative = np.diff(sigma_xy, 1)
        delta_sigma, list_start_end = utils.FindNegativeDerivativeIntervals(derivative, sigma_xy, file_created)
        if file_created:
            for i, element in enumerate(list_start_end):
                if i == len(list_start_end)-1:
                    file.write(f'{sigma_xy[element[0]]-sigma_xy[element[1]]}\n')    
                else:
                    file.write(f'{sigma_xy[element[0]]-sigma_xy[element[1]]} ')
        list_delta_sigma.append(delta_sigma)
    if file_created:
        file.close()
    return list_delta_sigma

