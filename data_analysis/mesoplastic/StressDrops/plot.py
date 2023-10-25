""" Every function that plots stress drops related data """

from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
import numpy as np
from mesoplastic.StressDrops import fit

rapport_path = '/home/victor/Documents/stage/rapport/'
# Every plot will have the same presentation (fontsize, labelsize, ...)
plt.style.use('/home/victor/Documents/stage/data_analysis/presentation.mplstyle')

def StressDropsPlot(filename: str,
                    list_time: np.ndarray,
                    list_sigma_xy: np.ndarray,
                    list_lambda: List[float],
                    ) -> Optional[Tuple[List[np.ndarray], List[np.ndarray]]]:
    """ Plots the Stress-Strain figure without the transient stage """
    plt.title('Stress drops study')
    plt.xlabel('time')
    plt.ylabel(r'$<\sigma_{xy}>$')
    for i, sigma_xy in enumerate(list_sigma_xy):
        plt.plot(list_time[:], sigma_xy[:], label=rf'$\lambda={list_lambda[i]}$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(rapport_path + filename + '.png')
    plt.show()
    return None

def DeltaSigmaDistribution(filename: str,
                           list_delta_sigma: List[List[float]],
                           list_lambda: List[float]
                           ):
    """ Plots the distribution of the value of the stress drops """
    x = np.linspace(1e-6, 1e-2, 1000)
    plt.title('Distribution of ' + r'$\Delta\sigma_{xy}$')
    plt.ylabel('Number of occurences')
    plt.xlabel('Value of stress drops')
    for i, delta_sigma in enumerate(list_delta_sigma):
        plt.hist(delta_sigma, label=rf'$\lambda$={list_lambda[i]}', log=True, fill=False, histtype='step', density=True,
                 bins=np.logspace(np.log10(1e-7), np.log10(1e-1), 1000))
    plt.loglog(x, np.power(x, -1.3)*0.01)
    plt.xscale('log')
    plt.legend(fontsize=30)
    plt.tight_layout()
    plt.savefig(rapport_path + filename + '.eps')
    plt.show()