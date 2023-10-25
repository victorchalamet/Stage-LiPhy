from mesoplastic.StressDrops import rawData, plot
from mesoplastic import compute, parse
import os
import numpy as np

def main():
    """ Main function """
    nb_points = 50000000
    list_dir, list_lambda = rawData.getRelevantStressDropsDirectories()
    if list_dir==None:
        return
    if not os.path.exists('/home/victor/Documents/stage/ComputedData/DeltaSigmaStressDrops.dat'):
        list_sigma_xy = rawData.getData(list_dir, nb_points)
        list_delta_sigma = compute.DeltaSigma(list_sigma_xy)
        # list_time = np.arange(200000, nb_points*0.01, 0.01)
        # plot.StressDropsPlot('SIZE128StressDrops', list_time, list_sigma_xy, list_lambda)
    else:
        list_delta_sigma = parse.ParseComputedDeltaSigma('/home/victor/Documents/stage/ComputedData/DeltaSigmaStressDrops.dat')
    plot.DeltaSigmaDistribution('SIZE128DeltaSigmaDistribution', list_delta_sigma, list_lambda)

if __name__ == '__main__':
    main()