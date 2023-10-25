""" General use functions """
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import os

def append(arr: np.ndarray, 
           value: Any,
           axis: int
           ) -> np.ndarray:
    """ Custom np.append function that work with empty array `arr` """
    if arr.size != 0:
        arr = np.append(arr, np.array([value]), axis=0)
    else:
        arr = np.append(arr, np.array([value]), axis)
    return arr

def getGdot(filename: str
            ) -> float:
    """ Search the value of `gamma dot` in the file's name """
    if 'Stress' in filename:
        gdot_indice_s = filename.find('gdot') + 4
        gdot_indice_e = filename.find('LX') - 1
    elif 'tran' in filename:
        gdot_indice_s = filename.find('GDOT') + 4
        gdot_indice_e = filename.find('Time') - 1
    else:
        gdot_indice_s = filename.find('GDOT') + 4
        gdot_indice_e = filename.find('Strain') - 1
    gdot = filename[gdot_indice_s:gdot_indice_e]
    return float(gdot)

def getLambda(filename: str
              ) -> float:
    """ Search the value of `lambda` in the file's name """
    if 'CONFIG' in filename:
        lambda_indice_s = filename.find('LAMBDA') + 6
        lambda_indice_e = filename.find('LX') - 1
        lambda_ = filename[lambda_indice_s:lambda_indice_e]
    else:
        lambda_indice_s = filename.find('LAMBDA') + 6
        lambda_indice_e = filename.find('SIZE') - 1
        lambda_ = filename[lambda_indice_s:lambda_indice_e]
    return float(lambda_)

def getSize(filename: str
            ) -> int:
    """ Search the value of the `system size` in the file's name """
    if 'tran' in filename:
        size_indice_s = filename.find('SIZE') + 4
        size_indice_e = filename.find('result') - 1
    else:
        size_indice_s = filename.find('LX') + 2
        size_indice_e = filename.find('GDOT')
    size = filename[size_indice_s:size_indice_e]
    return int(size)

def getTime(filename: str
            ) -> float:
    """ Search the value of the `time` in the file's name """
    time_indice_s = filename.find('Time') + 4
    time_indice_e = filename.find('.dat') - 1
    time = filename[time_indice_s:time_indice_e]
    return float(time)

def findLowestLen(list_file: List[str],
                  path: str
                  ) -> int:
    """ Find the lowest number of line in a list of data file """
    list_len_file = []
    for filename in list_file:
        file = os.path.join(path, filename)
        len_file = len(open(file, 'r').readlines())
        list_len_file.append(len_file)
    return min(list_len_file)

def findMaxTime(list_file: List[str]
                ) -> Optional[float]:
    """ Find the maximum time saved in a list of files """
    if len(list_file)==0:
        return None
    else:
        list_time = []
        for filename in list_file:
            list_time.append(getTime(filename))
        return max(list_time)

def orderPopt(list_popt:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """ Makes a list of popt for the flow curve fit """
    list_A = np.array([])
    list_n = np.array([])
    for indice in range(len(list_popt)):
        list_A = np.append(list_A, list_popt[indice][0])
        list_n = np.append(list_n, list_popt[indice][1])
    return list_A, list_n

def getVmax(value: np.ndarray,
            option: str,
            floor: float = 0.0
            ) -> Optional[int]:
    """ Gets the maximum indice at the end of which `value` fill a condition dictate by `option` (write in lowercase)
    # List of options:
    ### Lower
    First indice that verify `value < floor`.
    ### Relative
    First indice where the following `value` is greater than the current one.
    # Example:
    >>> value = np.array([3e-2, 5e-4, 8e-6, 2e-2])
    >>> vmax = getVmax(value, 1e-5, 'lower')
    >>> print(vmax)
    2
    """
    for i in range(value.size):
        if option=='lower':
            if value[i] < floor:
                return i
        elif option=='relative':
            if value[i] < value[i+1]:
                return i
    return None

def AccessElement(value_list: List[Any],
                  index_list: List[int]
                  ) -> List[Any]:
    """ Returns a new list from `value_list` with only indices from `index_list` """
    return [value_list[i] for i in index_list]

def getStrain(filename: str) -> float:
    """ Search the value of the `strain` in the file's name """
    size_indice_s = filename.find('Strain') + 6
    size_indice_e = filename.find('.dat') - 1
    size = filename[size_indice_s:size_indice_e]
    return float(size)

def Flatten(arr:np.ndarray) -> np.ndarray:
    """ Flattens the array such that it starts a new line with the end of the following and vice-versa.
    ### Example
    >>> Flatten([[1,2,3],[4,5,6]])
    [1,2,3,6,5,4] """
    result = np.array([])
    for i, element in enumerate(arr):
        if i%2==0:
            result = np.append(result, element)
        else:
            result = np.append(result, np.flipud(element))
    return np.asarray(result)

def getGMAX(filename:str) -> float:
    """ Search the value of `GMAX` in the file's name """
    size_indice_s = filename.find('GMAX') + 4
    size = filename[size_indice_s:]
    return float(size)

def get_color():
    """ Color generator """
    for item in ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'silver']:
        yield item

def FindNegativeDerivativeIntervals(derivative: np.ndarray,
                                    sigma_xy: List[float],
                                    file_created: bool) -> Tuple[List[float], List[Tuple[int, int]]]:
    list_start_end: List[Tuple[int, int]] = []
    delta_sigma: List[float] = []
    first = True
    start, end = 0, 0
    for i, value in enumerate(derivative):
        if value <= 0:
            if first:
                start = i
                following = i
                first = False
                # end = i+1
            if i == following:
                following += 1
                end = following
        if value > 0 and sigma_xy[start]-sigma_xy[end] > 0:
                if file_created:
                    list_start_end.append((start, end))
                delta_sigma.append(sigma_xy[start]-sigma_xy[end])
                first = True
                start, end = 0, 0
        # elif i == len(derivative)-1 and sigma_xy[start]-sigma_xy[i+1] > 0 and start != 0:
        #     list_start_end.append((start, i+1))
    return delta_sigma, list_start_end