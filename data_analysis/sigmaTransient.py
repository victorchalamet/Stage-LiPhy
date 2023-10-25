import sys
from mesoplastic.Transient import rawData
from mesoplastic.Transient import plot

""" Parameters """
fit_option = sys.argv[1] # power / fraction / two. fraction is the best one
system_size = sys.argv[2] # 128 / 1024

def main():
    """ Main function """
    list_dir, list_lambda = rawData.getRelevantNphysicDirectories(system_size)
    if list_dir==None:
        return
    all_x, all_sigma_xy = rawData.getData(list_dir)
    list_popt, list_pcov = plot.SigmaxyPlot('SIZE' + system_size + 'sigmaxyFlat', all_x, all_sigma_xy, list_lambda, fit_option)
    if len(list_popt) != 0 and len(list_lambda) > 1:
        if fit_option=='power':
            plot.SlopePlot('SIZE' + system_size + 'slopeLambda', list_lambda, list_popt)

        elif fit_option=='fraction':
            plot.ScreeningFractionPlot('SIZE' + system_size + 'ScreeningFractionCurve', list_lambda[1:7], list_popt[1:7], list_pcov[1:7])
        
        elif fit_option=='two':
            plot.ScreeningTwoPlot('SIZE' + system_size + 'ScreeningTwoCurve', list_lambda, list_popt, list_pcov)

if __name__ == '__main__':
    main()