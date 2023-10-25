""" Initialize the treatement of stress-strain related data """

from typing import Tuple, List, Dict
import os
import numpy as np
from mesoplastic import parse, utils
from mesoplastic.Strain import plot

def getRelevantPhysicDirectories(system_size: str
                                 ) -> Tuple[List[str], List[float]]:
    """ Finds all the useful directories for ploting flow curve """
    list_dir = []
    for directory in os.listdir('/home/victor/Documents/stage/'):
        if system_size in directory and 'Physic' in directory and '-1' in directory:
            list_dir.append(directory)
    list_dir, list_lambda = parse.OrderDirectory(list_dir, 'lambda')
    return list_dir, list_lambda

def getData(list_dir: List[str], list_lambda
            ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """ Gets data to plot the flow curve """
    all_gdot = []
    all_mean_sigma_xy = []
    all_SD_sigma_xy = []
    all_var_sigma_xy = []
    for i, directory in enumerate(list_dir):
        path = '/home/victor/Documents/stage/'+ directory + '/result/stress/'
        list_file = parse.OrganizeStressFile(os.listdir(path))
        if list_file==None:
            # If there's no data file goes to the next directory
            continue
        assert list_file is not None
        list_gdot = np.array([])
        # all_strain = []
        # all_sigma_xy = []
        mean_sigma_xy = np.array([])
        var_sigma_xy = np.array([])
        for filename in list_file:
            file = os.path.join(path, filename)
            list_strain, list_sigma_xy = parse.ParseStressFile(file)
            if len(list_strain) <= 300:
                # If the data file doesn't contain more than 300 lines goes to the next one
                continue
            list_gdot = np.append(list_gdot, np.array([utils.getGdot(filename)]))
            mean_sigma_xy = utils.append(mean_sigma_xy, list_sigma_xy[300:].mean(), 0)
            var_sigma_xy = utils.append(var_sigma_xy, list_sigma_xy[300:].var(), 0)
            # all_strain.append(list_strain)
            # all_sigma_xy.append(list_sigma_xy)
        all_gdot.append(list_gdot)
        all_mean_sigma_xy.append(mean_sigma_xy)
        all_SD_sigma_xy.append(np.sqrt(var_sigma_xy))
        all_var_sigma_xy.append(var_sigma_xy)
        # plot.SigmaxyStrainPlot(directory + 'SigmaxyAverage', all_strain, all_sigma_xy, list_gdot, list_lambda[i]) #type: ignore
    return all_gdot, all_mean_sigma_xy, all_var_sigma_xy, all_SD_sigma_xy

def orderGdot(all_gdot: List[np.ndarray]
              ) -> Dict[float, List[Tuple[int, int]]]:
    """ Makes a dictionnary with value of `gdot` as key and tuple of indices of the `all_gdot` list for value,
    sorted by ascending order using the key (gdot) """
    gdot_sorted: Dict[float, List[Tuple[int, int]]] = {}
    for i in range(len(all_gdot)):
        # Makes a dictionnary with value of gdot as key and tuple of indices of the all_gdot list for value
        for j in range(all_gdot[i].size):
            if all_gdot[i][j] not in gdot_sorted:
                gdot_sorted[all_gdot[i][j]] = [(i,j)]
            else:
                gdot_sorted[all_gdot[i][j]].append((i,j))

    copy = gdot_sorted.copy()
    for key in gdot_sorted.keys():
        if len(gdot_sorted[key]) <= 2:
            # If there's only 1 or 2 tuple for 1 value of gdot delete the key-value pair
            del copy[key]
    gdot_sorted = copy
    del copy
    # Here's the intermediate dictionnary sorted by ascending order using the key (gdot)
    return dict(sorted(gdot_sorted.items()))


def orderVarSigmaxy(all_var_sigma_xy: List[np.ndarray],
                    all_gdot: List[np.ndarray],
                    list_lambda: List[float]
                    ) -> Tuple[Dict[float, List[float]], List[List[float]]]:
    """ Based on the fact that the gdot's location in `all_gdot` corresponds to the `var_sigma_xy`'s location in `all_var_sigma_xy` """
    gdot_sorted = orderGdot(all_gdot)
    all_var_sigma_xy_sorted: Dict[float, List[float]] = {}
    for key in gdot_sorted.keys():
        # Makes a dictionnary as gdot_sorted but replace the tuple with the corresponding value of var_sigma_xy find in the all_var_sigma_xy list
        for j in range(len(gdot_sorted[key])):
            if key not in all_var_sigma_xy_sorted:
                all_var_sigma_xy_sorted[key] = [all_var_sigma_xy[gdot_sorted[key][j][0]][gdot_sorted[key][j][1]]]
            else:
                all_var_sigma_xy_sorted[key].append(all_var_sigma_xy[gdot_sorted[key][j][0]][gdot_sorted[key][j][1]])

    all_list_lambda = []
    for element in gdot_sorted.values():
        # Makes a list of lambda for each value of gdot
        indices = []
        for j in range(len(element)):
            indices.append(element[j][0])
        all_list_lambda.append(utils.AccessElement(list_lambda, indices))
    return all_var_sigma_xy_sorted, all_list_lambda