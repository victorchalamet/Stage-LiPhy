import sys
import numpy as np
from mesoplastic.Strain import plot, rawData

""" Parameters """
system_size = sys.argv[1]

def main():
    """ Main function """
    list_dir, list_lambda = rawData.getRelevantPhysicDirectories(system_size)
    all_gdot, all_mean_sigma_xy, all_var_sigma_xy, all_SD_sigma_xy = rawData.getData(list_dir, list_lambda)
    np.array(all_gdot, dtype=object)
    np.array(all_mean_sigma_xy, dtype=object)
    list_popt, list_pcov, yield_stress = plot.FlowCurvePlot('SIZE' + system_size + 'FlowCurveFlatFit',
                                                            all_mean_sigma_xy, all_gdot, list_lambda, all_SD_sigma_xy)
    # plot.FlowFitParameterPlot('SIZE' + system_size + 'FlowParameterFlat',
    #                           list_lambda, list_popt, yield_stress)
    # plot.VarSigmaxyPlot('SIZE' + system_size + 'VarCruveFlat',
    #                     all_var_sigma_xy, list_lambda, all_gdot)

if __name__ == '__main__':
    main()