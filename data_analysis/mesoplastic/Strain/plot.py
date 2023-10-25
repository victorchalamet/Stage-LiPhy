""" Every function that plots stress-strain related data """

from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from mesoplastic.Strain import fit, rawData
from mesoplastic import utils

rapport_path = '/home/victor/Documents/stage/rapport/'
# Every plot will have the same presentation (fontsize, labelsize, ...)
plt.style.use('/home/victor/Documents/stage/data_analysis/presentation.mplstyle')

def SigmaxyStrainPlot(filename: str,
                      all_strain: List[np.ndarray],
                      all_sigma_xy: List[np.ndarray],
                      list_gdot: List[float],
                      lambda_:float):
    """ Plots `sigma_xy_average` in terms of gamma for multiple values of gamma dot """
    plt.title(rf'$\lambda={lambda_}$')
    plt.xlabel(r'$\gamma$')
    plt.ylabel(r'$<\sigma_{xy}>$')
    for indice in range(len(all_strain)):
        plt.plot(all_strain[indice][300:400], all_sigma_xy[indice][300:400], label=r'$\dot{\gamma}=$'+f'{round(list_gdot[indice], 6)}')
    plt.legend()
    plt.savefig(rapport_path + filename + '.png')
    plt.show()

def FlowCurvePlot(filename: str,
                  all_mean_sigma_xy: np.ndarray,
                  all_gdot: np.ndarray,
                  list_lambda: List[float],
                  all_SD_sigma_xy: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Plots `mean_sigma_xy` in terms of gdot with a semilog scale for multiple values of lambda """
    # plt.title('Flow Curve for multiple values of ' + r'$\lambda$') #  + ' and their fit'
    plt.xlabel(r'$\dot{\gamma}$')
    plt.ylabel(r'$\overline{<\sigma_{xy}>}-\sigma_y$') # -\sigma_y
    colors = utils.get_color()
    list_popt = np.array([], ndmin=2)
    list_pcov = np.array([[[] for i in range(2)]])
    yield_stress = np.array([0.76165, 0.7287, 0.69276])
    for i in range(len(all_gdot)):
        mean_sigma_xy = all_mean_sigma_xy[i] - yield_stress[i]
        color = next(colors)
        plt.scatter(all_gdot[i][:], mean_sigma_xy, label=rf'$\lambda={list_lambda[i]}$', color=color, marker='s')
        # plt.errorbar(all_gdot[i], all_mean_sigma_xy[i], color=color, yerr=all_SD_sigma_xy[i])
        popt, pcov = fit.FlowCruveFit(all_gdot[i], mean_sigma_xy, yield_stress[i], color, [1 for i in range(2)], vmin=1)

        list_popt = utils.append(list_popt, popt, 1)
        list_pcov = utils.append(list_pcov, pcov, 2)

    leg = plt.legend(ncol=1, fontsize=45, loc=4)
    for handle in leg.legend_handles:
        handle.set_sizes([100.0])
    plt.tight_layout()
    plt.savefig(rapport_path + filename + '.eps')
    plt.show()
    return list_popt, list_pcov, yield_stress

def FlowFitParameterPlot(filename: str,
                         list_lambda: np.ndarray,
                         list_popt: np.ndarray,
                         yield_stress: np.ndarray):
    """ Plots the parameters of the fit in terms of lambda """
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
    plt.suptitle('Flow fit parameters in terms of ' + r'$\lambda$')
    # plt.title('Flow curve\'s fit parameter in terms of ' + r'$\lambda$')
    list_A, list_n = utils.orderPopt(list_popt)

    ax1.set_title(r'$\sigma_{y}$')
    ax1.set_xlabel(r'$\lambda$')
    ax1.plot(list_lambda, yield_stress)

    ax2.set_title('n')
    ax2.set_xlabel(r'$\lambda$')
    ax2.plot(list_lambda, list_n)
    ax2.plot(list_lambda, [1.0,1.0,1.0], 'r')

    # ax3.set_title('n')
    # ax3.set_xlabel(r'$\lambda$')
    # ax3.plot(list_lambda, list_n)

    # plt.ylabel('n')
    # plt.xlabel(r'$\lambda$')
    # plt.plot(list_lambda, list_n)
    # plt.plot(list_lambda, [1.0,1.0,1.0], 'r')

    plt.tight_layout()
    plt.savefig(rapport_path + filename + '.eps')
    plt.show()

def VarSigmaxyPlot(filename: str,
                   all_var_sigma_xy: List[np.ndarray],
                   list_lambda: List[float],
                   all_gdot: List[np.ndarray]):
    """ Plots the variance of `sigma_xy` in terms of lambda for multiple value of `gdot` """
    fig , (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})
    plt.suptitle('Variance of ' + r'$\sigma_{xy}$' + ' in terms of lambda')
    ax2.set_xlim(-0.025,0.125)
    ax2.set_title('Zoom on [0,0.1]')
    ax1.set_ylabel(r'$Var(\sigma_{xy})$')
    all_var_sigma_xy_dict, all_list_lambda = rawData.orderVarSigmaxy(all_var_sigma_xy, all_gdot, list_lambda)
    for ax in ax1, ax2:
        for i, key in enumerate(all_var_sigma_xy_dict.keys()):
            ax.plot(all_list_lambda[i], all_var_sigma_xy_dict[key], '*--', label=r'$\dot{\gamma}}$'+f'={key}')
        ax.label_outer()
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc=1, bbox_to_anchor=(.75,.87)) # Creates one legend for every subplots
        ax.set_xlabel(r'$\lambda$')
    plt.tight_layout()
    plt.savefig(rapport_path + filename + '.png')
    plt.show()