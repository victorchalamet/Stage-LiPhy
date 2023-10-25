""" Helps with parsing and organizing data """
from typing import List, Tuple, Optional, Dict
import numpy as np
from mesoplastic import utils

def ParseTransientFileSigma(path: str
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """ Parse the transient data file for `sigma_xy` and `x` only on y=`size//2 - 1` (middle) """
    time = utils.getTime(path)
    size = utils.getSize(path)
    file = open(path, 'r')
    sigma_xy_list = np.array([], dtype=float)
    list_x = np.array([], dtype=int)
    for line in file:
        line = line.split() # type: ignore
        if int(line[1]) == size//2 - 1:
            sigma_xy_list = np.append(sigma_xy_list, np.array([float(line[3])]))
            list_x = np.append(list_x, np.array([int(line[0])]))
    list_x = list_x[size//2:]-((size//2)-1) # center the list
    sigma_xy_list = sigma_xy_list[size//2:]-time
    file.close()
    return list_x, sigma_xy_list


def ParseStressFile(path: str
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """ Parse the stress data file for `strain` and `sigma_xy` """
    file = open(path, 'r').readlines()
    file = file[1:]
    strain = np.array([])
    list_sigma_xy_average = np.array([])
    for line in file:
        line = line.split() # type: ignore
        strain = np.append(strain, np.array([float(line[1])]))
        list_sigma_xy_average = np.append(list_sigma_xy_average, np.array([float(line[3])]))
    return strain, list_sigma_xy_average

def OrganizeTransientFilename(list_file: List[str]
                              ) -> Optional[str]:
    """ Only return the file with the max time """
    maxTime = utils.findMaxTime(list_file)
    if maxTime != None:
        for i, filename in enumerate(list_file):
            if 'Time'+str(maxTime) in filename:
                maxTimeIndice = i
        file = list_file[maxTimeIndice]
        return file
    else:
        return None

def OrganizeStressFile(list_file: List[str]
                       ) -> Optional[List[str]]:
    """ Sorts file by ascending value of gamma dot """
    list_file.sort(key=utils.getGdot)
    if len(list_file)==0:
        return None
    return list_file

def CreatePath(lambda_: str,
               system_size: str
               ) -> str:
    """ Creates a path depending on the value of lambda_ """
    if float(lambda_) >= 1:
        path = 'CONFIG_transient_TEST_DISTRIB_LAMBDA'+ lambda_ +'.00000000_LX'+ system_size +'GDOT1.00000000_Time200.00_.dat'
    else:
        while len(lambda_) < 10:
            lambda_ += '0'
        path = 'CONFIG_transient_TEST_DISTRIB_LAMBDA'+ lambda_ +'_LX'+ system_size +'GDOT1.00000000_Time200.00_.dat'
    return path

def OrderDirectory(list_dir: List[str],
                   option: str
                   ) -> Tuple[List[str], List[float]]:
    """ Orders the list of directory/file by ascending value of the `option`
    # List of option:
    - 'lambda'
    - 'time'
    - 'strain' """
    if option=='lambda':
        list_dir.sort(key=utils.getLambda)
        list_option = [utils.getLambda(filename) for filename in list_dir]
    elif option=='time':
        list_dir.sort(key=utils.getTime)
        list_option = [utils.getTime(filename) for filename in list_dir]
    elif option=='strain':
        list_dir.sort(key=utils.getStrain)
        list_option = [utils.getStrain(filename) for filename in list_dir]
    return list_dir, list_option

def ParseTransientFileAcc(path: str
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """ Parse the transient data file for `h_state` and `h_state_acc` """
    file = open(path, 'r')
    h_inst = np.array([], dtype=int)
    h_acc = np.array([], dtype=int)
    for line in file:
        line = line.split() # type: ignore
        h_inst = utils.append(h_inst, int(line[-4]), 0)
        h_acc = utils.append(h_acc, int(line[-3]), 0)
    h_inst = np.reshape(h_inst, (128,128))
    h_acc = np.reshape(h_acc, (128,128))
    file.close()
    return h_inst, h_acc

def ParseComputedCorrData(path: str
                          ) -> List[List[float]]:
    """ Parse the computed correlation data """
    all_h_acc: List[List[float]] = []
    with open(path, 'r') as file:
        for line in file:
            line = line.split() # type: ignore
            h_acc = [float(element) for element in line]
            all_h_acc.append(h_acc)
    return all_h_acc

def ParseComputedScreeningData(path: str
                               ) -> List[float]:
    """ Parse the computed typical screening length """
    with open(path, 'r') as file:
        for line in file:
            line = line.split() # type: ignore
            list_screening = [float(element) for element in line]
    return list_screening

def ParseStressDropsFile(path:str,
                    sigma_xy,
                    nb_points: int
                    ) -> None:
    """ Parse Stress-Strain file for stress drops data """
    with open(path, 'r') as file:
        first = True
        for i, line in enumerate(file):
            if i > nb_points:
                break
            if first:
                first = False
                continue
            line = line.split() # type: ignore
            sigma_xy[i-1] = float(line[0])

def ParseComputedDeltaSigma(path: str
                            ) -> List[List[float]]:
    """ Parse the computed delta sigma data """
    list_delta_sigma: List[List[float]] = []
    with open(path, 'r') as file:
        for line in file:
            line = line.split() # type: ignore
            delta_sigma = [float(element) for element in line]
            list_delta_sigma.append(delta_sigma)
    return list_delta_sigma
