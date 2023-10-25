""" Every function that plots activity distribution related data """

from typing import List, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
from mesoplastic.Distribution import fit
from mesoplastic import compute, utils

rapport_path = '/home/victor/Documents/stage/rapport/'
# Every plot will have the same presentation (fontsize, labelsize, ...)
plt.style.use('/home/victor/Documents/stage/data_analysis/presentation.mplstyle')

def SpatialCumulateDistributionPlot(filename: str,
                                    all_h_acc: List[np.ndarray],
                                    all_strain: np.ndarray):
    """ Plots the cumulate activity on multiple subplots for different times """
    fig, axs = plt.subplots(2, 6,sharex=True, sharey=True)
    plt.suptitle('Cumulate activity at different times')
    axs = axs.flat
    vmax = all_h_acc[-1].max()
    for j in range(len(all_h_acc)):
        im = axs[j].imshow(all_h_acc[j], vmin=0, vmax=vmax)
        axs[j].set_title(f'Strain={all_strain[j]}', fontsize=20)
        axs[j].axis('off')
    cax=fig.add_axes([0.93, 0.15, 0.03, 0.7]) # (Position of the left side, position of the bottom side, width, height)
    fig.colorbar(im, cax=cax)
    plt.tight_layout()
    plt.subplots_adjust(right=0.90)
    plt.savefig(rapport_path + filename + '.png')
    plt.show()

def SpatialDifferenceDistributionPlot(filename: str,
                                      all_h_acc: List[np.ndarray],
                                      all_strain: np.ndarray):
    """ Plots the difference of activity between 2 snapshots """
    fig, ax = plt.subplots(1, 1)
    plt.suptitle(f'Difference of activity between {all_strain[1]} and {all_strain[0]}')
    ax.axis('off')
    h_acc = all_h_acc[1] - all_h_acc[0]
    im = ax.imshow(h_acc, vmin=0, vmax=h_acc.max())
    cax=fig.add_axes([0.93, 0.15, 0.03, 0.7]) # (Position of the left side, position of the bottom side, width, height)
    fig.colorbar(im, cax=cax)
    plt.tight_layout()
    plt.savefig(rapport_path + filename + '.png')
    plt.show()

def SpatialAutoCorrelationPlot(filename: str,
                               all_h_acc: Union[List[List[List[np.ndarray]]],List[List[float]]],
                               list_lambda: List[float]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """ Plots the averaged spatial autocorrelation of the activity for multiple value of lambda """
    # plt.title('Spatial correlation of the cumulated activity. Strain window '+r'$\Delta\gamma=0.5$')
    plt.xlabel('x')
    plt.ylabel('C(x)')
    list_popt = []
    list_pcov = []
    colors = utils.get_color()
    list_x = np.array([i for i in range(1,11)])
    if isinstance(all_h_acc[0][0], float):
        for i, h_acc in enumerate(all_h_acc):
            assert isinstance(h_acc, List)
            color = next(colors)
            plt.scatter(list_x[1:9], h_acc[1:9], label=rf'$\lambda={list_lambda[i]}$', color=color)
            popt_coor, pcov_corr = fit.SpatialAutoCorrelationFit(list_x, h_acc, color, [50,1], vmin=1, vmax=9)
            list_popt.append(popt_coor)
            list_pcov.append(pcov_corr)
    else:
        for i, dir_h_acc in enumerate(all_h_acc):
            all_acorr_h_acc: List[np.ndarray] = []
            color = next(colors)
            for file_h_acc in dir_h_acc: # type: ignore
                h_acc = file_h_acc[1] - file_h_acc[0]
                assert isinstance(h_acc, np.ndarray)
                acorr_h_acc = compute.SpatialAutoCorrelation(h_acc)
                acorr_h_acc_x = [element[0] for element in acorr_h_acc]
                acorr_h_acc_y = acorr_h_acc[0]
                acorr_h_acc = np.asarray([(acorr_h_acc_x[i]+acorr_h_acc_y[i])/2 for i in range(len(acorr_h_acc_y))])
                all_acorr_h_acc.append(acorr_h_acc[1:11])
            acorr_h_acc = np.asarray(all_acorr_h_acc).mean(0)
            # vmax_fit = utils.getVmax(all_acorr_h_acc, 'relative')
            # vmax_curve = utils.getVmax(all_acorr_h_acc, 'lower', 1e-1)
            with open('/home/victor/Documents/stage/ComputedData/SpatialAutoCorrelationPlot.dat', 'a') as file:
                data = []
                for j, element in enumerate(acorr_h_acc):
                    if j != len(acorr_h_acc)-1:
                        data.append(f'{element} ')
                    else:
                        data.append(f'{element}\n')
                file.writelines(data)
            plt.semilogy(list_x, acorr_h_acc, label=rf'$\lambda={list_lambda[i]}$', color=color)
            popt_coor, pcov_corr = fit.SpatialAutoCorrelationFit(list_x, acorr_h_acc, color, [50,1], vmin=1, vmax=9)
            list_popt.append(popt_coor)
            list_pcov.append(pcov_corr)
    leg = plt.legend(fontsize=25, ncol=2, loc=8)
    for handle in leg.legend_handles:
        handle.set_sizes([100.0])
    plt.tight_layout()
    plt.savefig(rapport_path + filename + '.eps')
    plt.show()
    return list_popt, list_pcov

def ScreeningCorrPlot(filename: str,
                      list_popt: List[np.ndarray],
                      list_lambda: List[float]):
    """ Plots the typical correlation lenght in terms of lambda """
    # plt.title('Typical correlation lenght in terms of ' + r'$\lambda$')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('Typical correlation lenght' , fontsize=45)
    list_l = [element[1] for element in list_popt]
    plt.loglog(list_lambda, list_l, 's--')
    plt.tight_layout()
    plt.savefig(rapport_path + filename + '.eps')
    plt.show()