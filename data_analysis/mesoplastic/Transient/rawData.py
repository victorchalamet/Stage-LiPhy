""" Initialize the treatment of transient related data """

from typing import Optional, Tuple, List
import numpy as np
import os
from mesoplastic import parse, utils

def getRelevantNphysicDirectories(system_size: str
                                ) -> Optional[Tuple[List[str], List[float]]]:
    """ Find all the useful directories for plotting `sigma_xy` in terms of `x` """
    list_dir = []
    for directory in os.listdir('/home/victor/Documents/stage/'): # Iterate over every directories in /home/victor/Documents/stage/
        if system_size in directory and 'Nphysic' in directory: # Only wants none physic simulations
            list_dir.append(directory)
    if len(list_dir)==0:
        print("There's no such folder")
        return None
    list_dir, list_lambda = parse.OrderDirectory(list_dir, 'lambda') # Order by ascending value of lambda
    return list_dir, list_lambda

def getData(list_dir: List[str]
            ) -> Tuple[np.ndarray, np.ndarray]:
    """ Stores `sigma_xy` and `x` in arrays """
    all_x = np.array([[]])
    all_sigma_xy = np.array([[]])
    for directory in list_dir:
        path = '/home/victor/Documents/stage/' + directory + '/result/transient/'
        filename = parse.OrganizeTransientFilename(os.listdir(path)) # We only wants the last file saved
        if filename==None:
            continue
        assert filename is not None
        file = os.path.join(path, filename)
        list_x, sigma_xy_list = parse.ParseTransientFileSigma(file)
        all_x = utils.append(all_x, list_x, 1)
        all_sigma_xy = utils.append(all_sigma_xy, sigma_xy_list, 1)
    return all_x, all_sigma_xy