""" Initialize the treatement of activity distribution related data """

from typing import Tuple, Dict, List
from mesoplastic import utils, parse
import os
import numpy as np

def getOrderedFiles(system_size: str
                    ) -> Tuple[Dict[float, List[str]], List[str]]:
    """ Gets the ordered list of files for the spatial distribution data analysis """
    ordered_dict = {}
    list_dir = []
    list_indice = []
    list_path = []
    for directory in os.listdir('/home/victor/Documents/stage/'):
        if system_size in directory and 'GDOT' in directory:
            list_dir.append(directory)
    list_dir, list_lambda = parse.OrderDirectory(list_dir, 'lambda')
    for directory in list_dir:
        for i, element in enumerate(list_dir):
            if directory[:30] in element:
                if utils.getGMAX(directory) > utils.getGMAX(element):
                    list_indice.append(i)
    for i, indice in enumerate(list_indice):
        list_dir.pop(indice-i)
        list_lambda.pop(indice-i)

    for i, directory in enumerate(list_dir):
            list_file: List[str] = []
            path = '/home/victor/Documents/stage/' + directory + '/result/steady/'
            list_path.append(path)
            for filename in os.listdir(path):
                list_file.append(filename)
            list_file, list_strain = parse.OrderDirectory(list_file, 'strain')
            ordered_dict[list_lambda[i]] = list_file
    return ordered_dict, list_path

def getDifferenceData(ordered_dict: Dict[float, List[str]],
                      list_path: List[str],
                      step: int
                      ) -> List[List[List[np.ndarray]]]:
    all_h_acc: List[List[List[np.ndarray]]] = []
    for j, list_file in enumerate(ordered_dict.values()):
        all_h_inst = []
        file_h_acc = []
        dir_h_acc = []
        all_strain = np.array([])
        gdot = utils.getGdot(list_file[0])
        for i in range(len(list_file)):
            if i==0 or i+step+5>len(list_file):
                continue
            if i==np.asarray(list).any():
                new_list_file = utils.AccessElement(list_file, [-i-step-5,-i])
            else:
                new_list_file = utils.AccessElement(list_file, [-i-step,-i])
            for filename in new_list_file:
                strain = utils.getStrain(filename)
                file = os.path.join(list_path[j], filename)
                h_inst, h_acc = parse.ParseTransientFileAcc(file)
                all_strain = np.append(all_strain, strain)
                all_h_inst.append(h_inst)
                file_h_acc.append(h_acc)
                # plot.SpatialDifferenceDistributionPlot('LAMBDA'+LAMBDA+f'_GDOT{gdot}SpatialDistributionDifference', file_h_acc, all_strain)
            dir_h_acc.append(file_h_acc)
        all_h_acc.append(dir_h_acc)
    return all_h_acc