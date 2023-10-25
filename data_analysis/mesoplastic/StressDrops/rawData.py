""" Initialize the treatement of stress drops related data """

from typing import Tuple, List, Optional, Dict, Any
import os
import numpy as np
from mesoplastic import parse, utils
from multiprocessing import Process, Array, Pool, Lock
from collections import deque

def getRelevantStressDropsDirectories() -> Optional[Tuple[List[str], List[float]]]:
    """ Gets directories for the stress drops study """
    list_dir = []
    for directory in os.listdir('/home/victor/Documents/stage'):
        if 'Stress' in directory and 'Bad' not in directory:
            list_dir.append(directory)
    if len(list_dir)==0:
        print("There's no such folder")
        return None
    list_dir, list_lambda = parse.OrderDirectory(list_dir, 'lambda') # Order by ascending value of lambda
    return list_dir, list_lambda

def getData(list_dir: List[str],
            nb_points: int
            ) -> np.ndarray:
    """ Parse the stress drops files using the `multiprocessing` module """
    list_sigma_xy = np.array([[]])
    process_list: List[ParseMultiProcess] = []
    for i, directory in enumerate(list_dir):
        path = '/home/victor/Documents/stage/' + directory + '/result/stress/'
        filename = os.listdir(path)[0]
        file = os.path.join(path, filename)
        process_list.append(ParseMultiProcess(file, nb_points))
    
    for process in process_list:
        process.start()
    
    for process in process_list:
        process.join()
        sigma_xy = process.sigma_xy
        list_sigma_xy = utils.append(list_sigma_xy, sigma_xy, 1)
    return list_sigma_xy

class ParseMultiProcess(Process):
    """ Custom Class to use multiprocessing """
    def __init__(self,
                 file: str,
                 nb_points: int):
        Process.__init__(self)
        self.file = file
        self.nb_points = nb_points
        self.sigma_xy = Array('d', self.nb_points) # type: ignore 
    
    def run(self):
        """ Re write the `run` method of the `Process` parent class"""
        parse.ParseStressDropsFile(self.file, self.sigma_xy, self.nb_points)