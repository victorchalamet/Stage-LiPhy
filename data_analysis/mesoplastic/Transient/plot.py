""" Every function that plots transient related data """

from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import math
from mesoplastic.Transient import fit
from mesoplastic import utils, compute

rapport_path = '/home/victor/Documents/stage/rapport/'
# Every plot will have the same presentation (fontsize, labelsize, ...)
plt.style.use('/home/victor/Documents/stage/data_analysis/presentation.mplstyle')

def SigmaxyPlot(filename: str,
                all_x: np.ndarray, 
                all_sigma_xy: np.ndarray,
                list_lambda: List[float],
                option: str
                ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """ Plots `sigma_xy` in terms of x and their power/fraction plot for multiple values of lambda """
    # plt.title(f'System size = {all_x[0].size*2}') # . Data plot and their {option} fit
    plt.xlabel('x')
    plt.ylabel(r'$\sigma_{xy}$')
    # plt.loglog(all_x[0], fraction(all_x[0], 0.386, -1.917), 'b*', label=r'$\frac{x^{-4}}{b}$')
    colors = utils.get_color()
    list_popt: List[np.ndarray] = []
    list_pcov: List[np.ndarray] = []
    for i, list_x in enumerate(all_x):
        color = next(colors)
        normalized_all_sigma_xy = all_sigma_xy[i]/all_sigma_xy[i][1]
        plt.scatter(list_x[2:], normalized_all_sigma_xy[2:], marker='s', label=rf'$\lambda = {list_lambda[i]}$', color=color)
        vmax = utils.getVmax(normalized_all_sigma_xy, 'lower', 1e-5)
        if option=='power':
            popt, pcov = fit.SigmaxyPowerFit(list_x, normalized_all_sigma_xy, color, [100,-2], vmin=10, vmax=33)
        elif option=='fraction':
            if list_lambda[i] >= 5:
                vmin=2
            else:
                vmin=3
            popt, pcov = fit.SigmaxyFractionFit(list_x, normalized_all_sigma_xy, color, [1e-3, 1e-2], vmin=vmin, vmax=vmax)
        elif option=='two':
            list_indice_first = [[1,5],[1,5],[1,4],[1,4]]
            list_indice_second = [[40,90],[20,45],[15,28],[8,25]]
            popt, pcov = fit.SigmaxyTwoFit(list_x, normalized_all_sigma_xy, color, list_indice_first[i], list_indice_second[i], [[1,-2], [1,-4]], vmax=vmax) # type: ignore
        else:
            print("There's no such fit option")
            return list_popt, list_pcov

        list_pcov.append(pcov)
        list_popt.append(popt)

    plt.legend(loc=1, fontsize=30, ncol=1)
    plt.tight_layout()
    plt.savefig(rapport_path + option + 'fitted' + filename + '.eps') # + option + 'fitted'
    plt.show()
    return list_popt, list_pcov

def SlopePlot(filename: str,
              list_lambda: List[float],
              list_popt: List[np.ndarray]):
    """ Plots the slope of the fit in terms of lambda """
    list_slope = np.array([])
    for i, element in enumerate(list_popt):
        if element[1] != 0:
            list_slope = np.append(list_slope, element[1])
        else:
            list_lambda.pop(i)
    plt.title('Study of the evolution of the slope in terms of lambda')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('Slope')
    plt.scatter(list_lambda, list_slope, marker='o')
    plt.savefig(rapport_path + filename + '.png')
    plt.show()


def ScreeningFractionPlot(filename: str, 
                          list_lambda: List[float],
                          list_popt: List[np.ndarray],
                          list_pcov: List[np.ndarray]):
    """ Plots the typical value of screening in terms of lambda for the `fraction` fit """
    # plt.title('Typical screening length in terms of '+r'$\lambda$')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('Typical screening length ' + r'$\sqrt{\frac{a}{b}}$', fontsize=45)
    list_SD_distance = compute.ComputeSDDistance(list_popt, list_pcov, 'fraction')
    list_distance = np.array([np.sqrt(element[0]/element[1]) for element in list_popt])
    with open('ComputedData/ScreeningFractionPlot.dat', 'w') as file:
        data = []
        for element in list_distance:
            data.append(f'{element} ')
        file.writelines(data)
    # popt_screening, pcov_screening = curve_fit(fit.powerFit, list_lambda, list_distance)
    # corr = compute.getRsquared(list_lambda, list_distance, popt_screening, fit.powerFit)
    plt.loglog(list_lambda, list_distance, '--', color='b')
    plt.scatter(list_lambda, list_distance, s=100, color='b', marker='s')
    # plt.errorbar(list_lambda, list_distance, yerr=list_SD_distance, ls="None", color='b')
    # fit.ScreeningFit(list_lambda, list_distance)
    plt.tight_layout()
    plt.savefig(rapport_path + filename + '.eps')
    plt.show()
    
def ScreeningTwoPlot(filename: str,
                     list_lambda: List[float],
                     list_popt: List[np.ndarray],
                     list_pcov: List[np.ndarray]):
    """ Plots the typical value of screening in terms of lambda for the `two` fit """
    plt.title('Typical value of screening in terms of lambda')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('Typical value of screening ' + r'$\sqrt{\frac{b}{a}}$')
    list_distance = [math.sqrt(element[1][0]/element[0][0]) for element in list_popt]
    list_SD_distance = compute.ComputeSDDistance(list_popt, list_pcov, 'two')
    list_lambda = list_lambda[:4]
    plt.loglog(list_lambda[1:], list_distance[1:], 's', color='b')
    plt.errorbar(list_lambda[1:], list_distance[1:], yerr=list_SD_distance[1:], linestyle='--', color='b')
    plt.savefig(rapport_path + filename + '.png')
    plt.show()