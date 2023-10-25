from mesoplastic import parse, utils
from mesoplastic.Strain.fit import YieldStressFit
from mesoplastic.Transient import rawData, fit
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use('/home/victor/Documents/stage/data_analysis/presentation.mplstyle')

def linear(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a*x + b

def expFit(x: np.ndarray, a: float, l: float) -> np.ndarray:
    return a * np.exp(-x/l)

def EdgeEffects() -> None:
    """ Shows the finite size effects """
    list_file = [
        '/home/victor/Documents/stage/NphysicLAMBDA0_SIZE128/result/transient/CONFIG_transient_TEST_DISTRIB_LAMBDA0.00000000_LX128GDOT1.00000000_Time200.00_.dat',
        '/home/victor/Documents/stage/NphysicLAMBDA0_SIZE256/result/transient/CONFIG_transient_TEST_DISTRIB_LAMBDA0.00000000_LX256GDOT1.00000000_Time2.00_.dat',
        '/home/victor/Documents/stage/NphysicLAMBDA0_SIZE512/result/transient/CONFIG_transient_TEST_DISTRIB_LAMBDA0.00000000_LX512GDOT1.00000000_Time2.00_.dat',
        '/home/victor/Documents/stage/NphysicLAMBDA0_SIZE1024/result/transient/CONFIG_transient_TEST_DISTRIB_LAMBDA0.00000000_LX1024GDOT1.00000000_Time2.00_.dat',
        '/home/victor/Documents/stage/NphysicLAMBDA0_SIZE2048/result/transient/CONFIG_transient_TEST_DISTRIB_LAMBDA0.00000000_LX2048GDOT1.00000000_Time2.00_.dat'
    ]
    plt.figure()
    plt.title(r'$\sigma_{xy}$' + ' in terms of x for multiple system size')
    plt.xlabel('x')
    plt.ylabel(r'$\sigma_{xy}$')
    for filename in list_file:
        list_x, list_sigma_xy = parse.ParseTransientFileSigma(filename)
        size = utils.getSize(filename)

        plt.loglog(list_x, list_sigma_xy/list_sigma_xy[0], label=f'System size={size}')
    plt.legend(fontsize=30)
    plt.tight_layout()
    plt.savefig('/home/victor/Documents/stage/rapport/edgeEffects.eps')
    plt.show()

def SpecialCurve():
    """ Plots the 2nd point of the CONFIG_transient data in terms of lambda """
    system_size = '1024'
    all_sigma_xy = []
    list_dir = []
    for directory in os.listdir('/home/victor/Documents/stage/'):
        if system_size in directory and 'Nphysic' in directory:
            list_dir.append(directory)
    if len(list_dir)==0:
        print("There's no such folder")
        return
    list_dir, list_lambda = parse.orderDirectory(list_dir, 'lambda')
    for directory in list_dir:
        path = '/home/victor/Documents/stage/' + directory + '/result/transient/'
        filename = parse.organizeTransientFilename(os.listdir(path))
        if filename==None:
            continue
        # Find the value of lambda in the name of the file and store it in an array
        file = os.path.join(path, filename)

        # Gets the list of position and of sigma_xy of the file
        list_x, sigma_xy_list = parse.parseTransientFileSigma(file)
        all_sigma_xy.append(sigma_xy_list[1:2])
    
    all_gdot = []
    all_mean_sigma_xy = []
    list_dir = []
    system_size = '128'
    for directory in os.listdir('/home/victor/Documents/stage/'):
        if system_size in directory and 'Physic' in directory:
            list_dir.append(directory)
    list_dir, list_lambda = parse.orderDirectory(list_dir, 'lambda')
    for i, directory in enumerate(list_dir):
        path = '/home/victor/Documents/stage/'+ directory + '/result/stress/'
        list_file = parse.organizeStressFilename(os.listdir(path))
        if list_file==None:
            continue
        list_gdot = np.array([])
        mean_sigma_xy = np.array([])
        for filename in list_file:
            file = os.path.join(path, filename)
            list_strain, list_sigma_xy = parse.parseStressFile(file)

            if len(list_strain) <= 300:
                continue

            list_gdot = np.append(list_gdot, np.array([utils.getGdot(filename)]))
            mean_sigma_xy = utils.append(mean_sigma_xy, list_sigma_xy[300:].mean(), 0)
        all_gdot.append(list_gdot)
        all_mean_sigma_xy.append(mean_sigma_xy)

    np.array(all_gdot, dtype=object)
    np.array(all_mean_sigma_xy, dtype=object)

    yield_stress = np.array([])
    for i in range(len(all_gdot)):
        popt_sigma, pcov_sigma = curve_fit(YieldStressFit, all_gdot[i], all_mean_sigma_xy[i], [1,1,1])
        yield_stress = np.append(yield_stress, popt_sigma[0])

    yield_stress = utils.AccessElement(yield_stress, [1,4,6,7])
    all_sigma_xy = utils.AccessElement(all_sigma_xy, [0,4,6,7])

    list_sigma_xy = []
    for element in all_sigma_xy:
        list_sigma_xy.append(element[0])

    popt, pcov = curve_fit(linear, list_sigma_xy, yield_stress)
    corr1 = utils.getRsquared(np.array(list_sigma_xy), yield_stress, popt, linear) 
    corr2 = abs(pcov[0][1]/np.sqrt(pcov[0][0]*pcov[1][1]))**2
    plt.figure()
    plt.title('Yield stress in terms of the 2nd point of the CONFIG_transient file')
    plt.xlabel('2nd point of the curve')
    plt.ylabel(r'$\sigma_{y}$')
    plt.plot(list_sigma_xy, yield_stress)
    plt.plot(list_sigma_xy, linear(np.array(list_sigma_xy), *popt), '*--', color='r', label=f'a={round(popt[0],3)}, b={round(popt[1],3)}\n'+r'$r^{2}$'+f'={round(corr1, 3)} ou {round(corr2, 3)}\n{pcov}')
    plt.legend(fontsize=15)
    plt.savefig('/home/victor/Documents/stage/rapport/yieldStressCorr.png')
    plt.show()


def Figure1():
    """ Plots the first figure of my  """
    path = '/home/victor/Documents/stage/NphysicLAMBDA0.1_SIZE128/result/transient/CONFIG_transient_TEST_DISTRIB_LAMBDA0.10000000_LX128GDOT1.00000000_Time200.00_.dat'
    time = utils.getTime(path)
    file = open(path, 'r')
    sigma_xy_list = np.array([], dtype=float)
    for line in file:
        line = line.split()
        sigma_xy_list = np.append(sigma_xy_list, np.array([float(line[3])]))
    sigma_xy_list -= time-198
    # sigma_xy_list = np.log(sigma_xy_list)
    sigma_xy_list = sigma_xy_list.reshape((128,128))

    fig = plt.figure()
    ax1 = plt.subplot(221)
    ax1.set_xlim(53,73)
    ax1.set_ylim(53,73)
    im = ax1.imshow(sigma_xy_list)
    divider = make_axes_locatable(ax1)
    cb_ax = divider.append_axes('left', size='5%', pad=0.05)
    fig.colorbar(im, cax=cb_ax)
    im.figure.axes[1].tick_params(axis='y', labelsize=20)
    ax1.axis('off')



    ax3 = plt.subplot(212)
    ax3.set_xlabel('x')
    ax3.set_ylabel(r'$\sigma_{xy}$')
    list_dir = ['NphysicLAMBDA0_SIZE1024', 'NphysicLAMBDA0.01_SIZE1024', 'NphysicLAMBDA0.1_SIZE1024', 'NphysicLAMBDA0.3_SIZE1024',
                'NphysicLAMBDA1_SIZE1024', 'NphysicLAMBDA5_SIZE1024', 'NphysicLAMBDA10_SIZE1024', 'NphysicLAMBDA50_SIZE1024']
    list_lambda = [0.0, 0.01, 0.1, 0.3, 1.0, 5.0, 10.0, 50.0]
    colors = utils.get_color()
    all_x, all_sigma_xy = rawData.getData(list_dir)
    for i, list_x in enumerate(all_x):
        color = next(colors)
        normalized_all_sigma_xy = all_sigma_xy[i]/all_sigma_xy[i][1]
        ax3.scatter(list_x[2:], normalized_all_sigma_xy[2:], marker='s', label=rf'$\lambda = {list_lambda[i]}$', color=color)
        vmax = utils.getVmax(normalized_all_sigma_xy, 'lower', 1e-5)
        if list_lambda[i] >= 5:
            vmin=2
        else:
            vmin=3
        fit.SigmaxyFractionFit(list_x, normalized_all_sigma_xy, color, ax3, [1e-3, 1e-2], vmin=vmin, vmax=vmax)

    ax2 = plt.subplot(222)
    list_screening = parse.ParseComputedScreeningData('ComputedData/ScreeningFractionPlot.dat')
    # list_screening = utils.AccessElement(list_screening, [0,1,3,5])
    ax2.set_xlabel(r'$\lambda$')
    ax2.set_ylabel(r'$\sqrt{\frac{a}{b}}$')
    ax2.loglog(list_lambda[1:7], list_screening, '--', color='b')
    ax2.scatter(list_lambda[1:7], list_screening, s=100, color='b', marker='s')
    ax2.set_box_aspect(1)

    plt.tight_layout()
    # plt.subplots_adjust(right=0.926)
    plt.savefig('/home/victor/Documents/stage/rapport/ColorMapSigmaxy.png')
    plt.show()


def LinkScreeningCorr():
    all_h_acc = parse.ParseComputedCorrData('ComputedData/SpatialAutoCorrelationPlot.dat')
    list_lambda = [0.01, 0.1, 1.0, 10.0]
    list_x = np.array([i for i in range(1,11)])
    list_popt = []
    list_pcov = []
    for h_acc in all_h_acc:
        popt_corr, pcov_corr = curve_fit(expFit, list_x[1:9], h_acc[1:9], [50,1])
        list_popt.append(popt_corr)
        list_pcov.append(pcov_corr)
    list_corr = [element[1] for element in list_popt[2:]]

    list_screening = parse.ParseComputedScreeningData('ComputedData/ScreeningFractionPlot.dat')
    list_screening = utils.AccessElement(list_screening, [0,1,3,5])

    plt.figure()
    plt.title('Link between screening and correlation')
    plt.ylabel('Correlation length')
    plt.xlabel('Screening length')
    plt.text(1.4, 0.5, r'$\lambda=10$', fontsize=30)
    plt.text(5, 2, r'$\lambda=1$', fontsize=30)
    plt.text(14, 3.6, r'$\lambda=0.1$', fontsize=30)
    plt.text(30, 5.2, r'$\lambda=0.01$', fontsize=30)
    plt.plot(list_screening, list_corr, 's--')
    plt.tight_layout()
    plt.savefig('/home/victor/Documents/stage/rapport/LinkScreeningCorr.eps')
    plt.show()

def PlasticActivityMaps():
    """ Plots the activity with a color map for multiple value of lambda and of delta gamma """
    List_file = [
        ['/home/victor/Documents/stage/PhysicLAMBDA0_SIZE128_GDOT0.0001_GMAX6/result/steady/CONFIG_steady_state_TEST_DISTRIB_LAMBDA0.00000000_LX128GDOT0.00010000_Strain8.00_.dat',
         '/home/victor/Documents/stage/PhysicLAMBDA0_SIZE128_GDOT0.0001_GMAX6/result/steady/CONFIG_steady_state_TEST_DISTRIB_LAMBDA0.00000000_LX128GDOT0.00010000_Strain7.50_.dat',
         '/home/victor/Documents/stage/PhysicLAMBDA0_SIZE128_GDOT0.0001_GMAX6/result/steady/CONFIG_steady_state_TEST_DISTRIB_LAMBDA0.00000000_LX128GDOT0.00010000_Strain6.00_.dat',
         '/home/victor/Documents/stage/PhysicLAMBDA0_SIZE128_GDOT0.0001_GMAX6/result/steady/CONFIG_steady_state_TEST_DISTRIB_LAMBDA0.00000000_LX128GDOT0.00010000_Strain3.00_.dat'],
        ['/home/victor/Documents/stage/PhysicLAMBDA0.01_SIZE128_GDOT0.0001_GMAX10/result/steady/CONFIG_steady_state_TEST_DISTRIB_LAMBDA0.01000000_LX128GDOT0.00010000_Strain12.00_.dat',
          '/home/victor/Documents/stage/PhysicLAMBDA0.01_SIZE128_GDOT0.0001_GMAX10/result/steady/CONFIG_steady_state_TEST_DISTRIB_LAMBDA0.01000000_LX128GDOT0.00010000_Strain11.50_.dat',
          '/home/victor/Documents/stage/PhysicLAMBDA0.01_SIZE128_GDOT0.0001_GMAX10/result/steady/CONFIG_steady_state_TEST_DISTRIB_LAMBDA0.01000000_LX128GDOT0.00010000_Strain10.00_.dat',
          '/home/victor/Documents/stage/PhysicLAMBDA0.01_SIZE128_GDOT0.0001_GMAX10/result/steady/CONFIG_steady_state_TEST_DISTRIB_LAMBDA0.01000000_LX128GDOT0.00010000_Strain7.00_.dat'],
        ['/home/victor/Documents/stage/PhysicLAMBDA0.1_SIZE128_GDOT0.0001_GMAX10/result/steady/CONFIG_steady_state_TEST_DISTRIB_LAMBDA0.10000000_LX128GDOT0.00010000_Strain12.00_.dat',
          '/home/victor/Documents/stage/PhysicLAMBDA0.1_SIZE128_GDOT0.0001_GMAX10/result/steady/CONFIG_steady_state_TEST_DISTRIB_LAMBDA0.10000000_LX128GDOT0.00010000_Strain11.50_.dat',
          '/home/victor/Documents/stage/PhysicLAMBDA0.1_SIZE128_GDOT0.0001_GMAX10/result/steady/CONFIG_steady_state_TEST_DISTRIB_LAMBDA0.10000000_LX128GDOT0.00010000_Strain10.00_.dat',
          '/home/victor/Documents/stage/PhysicLAMBDA0.1_SIZE128_GDOT0.0001_GMAX10/result/steady/CONFIG_steady_state_TEST_DISTRIB_LAMBDA0.10000000_LX128GDOT0.00010000_Strain7.00_.dat']
          ]
    delta_gamma = [0.5, 2, 5]
    list_lambda = [0, 0.01, 0.1]
    all_h_acc = []
    for list_file in List_file:
        file_h_acc = []
        for file in list_file:
            h_inst, h_acc = parse.ParseTransientFileAcc(file)
            file_h_acc.append(h_acc)
        all_h_acc.append(file_h_acc)

    new_all_h_acc = []
    for file_h_acc in all_h_acc:
        new_file_h_acc = []
        for i, h_acc in enumerate(file_h_acc):
            if i==0:
                continue
            new_file_h_acc.append(file_h_acc[0] - h_acc)
        new_all_h_acc.append(new_file_h_acc)

    list_vmax = [[9,15,26],
                 [9,16,20],
                 [9,11,19]]


    fig, axs = plt.subplots(3,3, sharex=True, sharey=True, figsize=(19.2, 15.5))
    # change aspect ratio
    # fig.suptitle('Cumulated activity maps for multiple values of '+r'$\lambda$'+' and for multiple values of '+r'$\Delta\gamma$')
    list_im = []
    for i, ax in enumerate(axs):
        for j, a in enumerate(ax):
            im = a.imshow(new_all_h_acc[i][j], vmin=0, vmax=list_vmax[i][j])
            if i==0:
                a.set_title(rf'$\Delta\gamma={delta_gamma[j]}$')
                list_im.append(im)
            if j==0:
                a.set_ylabel(rf'$\lambda={list_lambda[i]}$')
            a.grid(False)
            a.tick_params(
                bottom=False,
                left=False,
                labelbottom=False,
                labelleft=False
            )
    list_cax = []
    list_cax.append(fig.add_axes([0.31, 0.01, 0.03, 0.95]))
    list_cax.append(fig.add_axes([0.63, 0.01, 0.03, 0.95]))
    list_cax.append(fig.add_axes([0.945, 0.01, 0.03, 0.95])) # (Position of the left side, position of the bottom side, width, height)
    for i, im in enumerate(list_im):
        # im.set_clim(vmin=0, vmax=list_vmax[i])
        cbar = fig.colorbar(im, cax=list_cax[i])
        # cbar.mappable.set_clim(0, list_vmax[i])
    plt.tight_layout()
    plt.subplots_adjust(left=0)
    plt.savefig('/home/victor/Documents/stage/rapport/PlasticActivityMaps.png')
    plt.show()

if __name__ == '__main__':
    Figure1()