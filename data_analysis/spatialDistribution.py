from typing import List, Dict
import sys
import os
from mesoplastic.Distribution import plot, rawData
from mesoplastic import parse, utils
import numpy as np

""" Parameters """
system_size = sys.argv[1]
option = sys.argv[2]
if option == 'difference':
    step = int(sys.argv[3])
if option == 'cumulate':
    LAMBDA = sys.argv[3]
    # GMAX = sys.argv[4]


def Snapshots(ordered_dict: Dict[float, List[str]],
             list_path: List[str]):
    """ Main function for snapshots data """
    for i, key in enumerate(ordered_dict.keys()):
        if key == float(LAMBDA):
            list_file = ordered_dict[key]
            path = list_path[i]
    all_h_inst = []
    all_h_acc = []
    all_strain = np.array([])
    h_inst_init, h_acc_init = parse.ParseTransientFileAcc(path+list_file[0])
    list_file = list_file[1:]
    for filename in list_file:
        strain = utils.getStrain(filename)
        gdot = utils.getGdot(filename)
        file = os.path.join(path, filename)
        h_inst, h_acc = parse.ParseTransientFileAcc(file)
        h_inst -= h_inst_init
        h_acc -= h_acc_init
        all_strain = np.append(all_strain, strain)
        all_h_inst.append(h_inst)
        all_h_acc.append(h_acc)

    plot.SpatialCumulateDistributionPlot('LAMBDA'+LAMBDA+f'_GDOT{gdot}SpatialDistributionCumulate', all_h_acc, all_strain)

def Difference(ordered_dict: Dict[float, List[str]],
               list_path: List[str],
               step: int):
    """ Main function for cumulated data between two snapshots """
    list_lambda = list(ordered_dict.keys())
    if os.path.exists('/home/victor/Documents/stage/ComputedData/SpatialAutoCorrelationPlot.dat'):
        all_h_acc_compute = parse.ParseComputedCorrData('/home/victor/Documents/stage/ComputedData/SpatialAutoCorrelationPlot.dat')
        list_popt, list_pcov = plot.SpatialAutoCorrelationPlot('SIZE'+ system_size + f'STEP{step}' + 'AutoCorrelation', all_h_acc_compute, list_lambda)
    else:
        all_h_acc = rawData.getDifferenceData(ordered_dict, list_path, step)
        list_popt, list_pcov = plot.SpatialAutoCorrelationPlot('SIZE'+ system_size + f'STEP{step}' + 'AutoCorrelation', all_h_acc, list_lambda)
    plot.ScreeningCorrPlot('SIZE'+ system_size + f'STEP{step}' + 'ScreeningCorr', list_popt[1:], list_lambda[1:])
        
if __name__ == '__main__':
    ordered_dict, list_path = rawData.getOrderedFiles(system_size)
    if option == 'difference':
        Difference(ordered_dict, list_path, step)
    elif option == 'cumulate':
        Snapshots(ordered_dict, list_path)