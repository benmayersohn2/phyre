import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator, FuncFormatter
import copy
import waipy
from phyre.analysis import analysis as al
from phyre import helpers
from typing import Dict, List, Tuple, Union
from phyre import constants as c


def fft(data):
    """FFT spectrum
    """
    n = len(data)
    X = np.fft.fft(data)
    sxx = ((X * np.conj(X)) / (n))
    f = -np.fft.fftfreq(n)[int(np.ceil(n / 2.)):]
    sxx = np.abs(sxx)
    sxx = sxx[int(np.ceil(n / 2.)):]
    return f, sxx


def levels(result, dtmin):
    """
    Power levels
    """

    dtmax = result['power'].max()
    lev = []
    for i in range(int(np.log2(dtmax / dtmin)) + 1):
        dtmin = dtmin * 2
        lev.append(dtmin)
    return lev


def wavelet_plot(var, time, data, dtmin, result, **kwargs):
    """
    PLOT WAVELET TRANSFORM
    var = title name from data
    time  = vector get in load function
    data  = from normalize function
    dtmin = minimum resolution :1 octave
    result = dict from cwt function
    kwargs:
        no_plot
        filename
        xlabel_cwt
        ylabel_cwt
        ylabel_data
        plot_phase : bool, defaults to False
    """
    # frequency limit
    # print result['period']
    # lim = np.where(result['period'] == result['period'][-1]/2)[0][0]
    # """Plot time series """

    fig = plt.figure(figsize=(15, 10), dpi=300)

    gs1 = gridspec.GridSpec(4, 3)
    gs1.update(left=0.07, right=0.7, wspace=0.5, hspace=0, bottom=0.15, top=0.97)

    ax1 = plt.subplot(gs1[0, :])
    ax1.xaxis.set_visible(False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2 = plt.subplot(gs1[1:4, :])  # , axisbg='#C0C0C0')

    gs2 = gridspec.GridSpec(4, 1)
    gs2.update(left=0.7, right=0.98, hspace=0, bottom=0.15, top=0.97)
    ax5 = plt.subplot(gs2[1:4, 0], sharey=ax2)
    plt.setp(ax5.get_yticklabels(), visible=False)

    gs3 = gridspec.GridSpec(6, 1)
    gs3.update(
        left=0.77, top=0.97, right=0.98, hspace=0.6, wspace=0.01,
    )
    ax3 = plt.subplot(gs3[0, 0])

    ax1.plot(time, data)
    ax1.axis('tight')
    ax1.set_xlim(time.min(), time.max())
    ax1.set_ylabel(kwargs.get('ylabel_data', 'Amatplotlibitude'), fontsize=15)
    ax1.set_title('%s' % var, fontsize=17)
    ax1.yaxis.set_major_locator(MaxNLocator(prune='lower'))
    ax1.grid(True)
    ax1.xaxis.set_visible(False)
    joint_wavelet = result['joint_wavelet']
    wavelet_x = np.arange(-result['nw'] / 2, result['nw'] / 2)
    ax3.plot(
        wavelet_x,
        joint_wavelet.real,
        'k',
        label='Real part'
    )
    ax3.plot(
        wavelet_x,
        joint_wavelet.imag,
        '--k',
        label='Imag part'
    )
    ax3.plot(
        wavelet_x,
        result['mean_wavelet'],
        'g',
        label='Mean'
    )

    # try to infer the xlims by selecting the limit at 5% of maximum value of
    # real part
    limit_index = np.where(
        np.abs(joint_wavelet.real) > 0.05 * np.max(np.abs(joint_wavelet.real))
    )
    ax3.set_xlim(-wavelet_x[limit_index[0][0]], wavelet_x[limit_index[0][0]])
    # ax3.axis('tight')
    # ax3.set_xlim(-100, 100)
    # ax3.set_ylim(-0.3,0.3)
    # ax3.set_ylim(
    #     [np.min(result['joint_wavelet']),np.max(result['joint_wavelet'])])
    ax3.set_xlabel('Time', fontsize=10)
    ax3.set_ylabel('Amplitude', fontsize=10)
    ax3.set_title(r'$\psi$ (t/s) {0} in time domain'.format(result['mother']))
    # ------------------------------------------------------------------------#
    # ax4.plot(result['ondaleta'],'k')
    # ax4.set_xlabel('Frequency', fontsize=10)
    # ax4.set_ylabel('Amplitude', fontsize=10)
    # ax4.set_title('$\psi^-$  Frequency domain', fontsize=13)
    # ------------------------------------------------------------------------#
    # colorbar location
    position = fig.add_axes([0.07, 0.07, 0.6, 0.01])

    plot_phase = kwargs.get('plot_phase', False)
    if plot_phase:
        phases = np.arctan(
            np.imag(result['wave']),
            np.real(result['wave'])
        )
        # import IPython
        # IPython.embed()
        # exit()
        phase_levels = np.linspace(phases.min(), phases.max(), 10)
        norm = colors.DivergingNorm(vcenter=0)
        pc = ax2.contourf(
            time,
            np.log2(result['period']),
            phases,
            phase_levels,
            cmap=plt.cm.get_cmap('seismic'),
            norm=norm
        )
        cbar = plt.colorbar(
            pc,
            cax=position,
            orientation='horizontal',
        )
        cbar.set_label('Phase [rad]')

    else:
        # """ Contour plot wavelet power spectrum """
        lev = levels(result, dtmin)
        # import IPython
        # IPython.embed()
        # exit()
        cmap = copy.copy(plt.cm.get_cmap('viridis'))
        cmap.set_over('yellow')
        cmap.set_under('cyan')
        cmap.set_bad('red')
        # ax2.imshow(np.log2(result['power']), cmap='jet', interpolation=None)
        # ax2.set_aspect('auto')
        pc = ax2.contourf(
            time,
            np.log2(result['period']),
            np.log2(result['power']),
            np.log2(lev),
            cmap=cmap,
        )
        # print(time.shape)
        # print(np.log2(result['period']).shape)
        # print(np.log2(result['power']).shape)
        # X, Y = np.meshgrid(time, np.log2(result['period']))
        # ax2.scatter(
        #     X.flat,
        #     Y.flat,
        # )

        # 95% significance contour, levels at -99 (fake) and 1 (95% signif)
        pc2 = ax2.contour(
            time,
            np.log2(result['period']),
            result['sig95'],
            [-99, 1],
            linewidths=2
        )
        ax2.plot(time, np.log2(result['coi']), 'k')
        # cone-of-influence , anything "below"is dubious
        ax2.fill_between(
            time,
            np.log2(result['coi']),
            int(np.log2(result['period'][-1]) + 1),
            # color='white',
            alpha=0.6,
            hatch='/'
        )

        def cb_formatter(x, pos):
            # x is in base 2
            linear_number = 2 ** x
            return '{:.1f}'.format(linear_number)

        cbar = plt.colorbar(
            pc, cax=position, orientation='horizontal',
            format=FuncFormatter(cb_formatter),
        )
        cbar.set_label('Power')

    yt = range(
        int(np.log2(result['period'][0])),
        int(np.log2(result['period'][-1]) + 1)
    )  # create the vector of periods
    Yticks = [float(2 ** p) for p in yt]  # make 2^periods
    # Yticks = [int(i) for i in Yticks]
    ax2.set_yticks(yt)
    ax2.set_yticklabels(Yticks)
    ax2.set_ylim(
        ymin=(np.log2(np.min(result['period']))),
        ymax=(np.log2(np.max(result['period'])))
    )
    ax2.set_ylim(ax2.get_ylim()[::-1])
    ax2.set_xlabel(kwargs.get('xlabel_cwt', 'Time'), fontsize=12)
    ax2.set_ylabel(kwargs.get('ylabel_cwt', 'Period'), fontsize=12)

    # if requested, limit the time range that we show
    xmin = kwargs.get('xmin', None)
    xmax = kwargs.get('xmax', None)
    if xmin is not None or xmax is not None:
        for ax in (ax1, ax2):
            ax.set_xlim(xmin, xmax)

    # Plot global wavelet spectrum
    f, sxx = fft(data)
    ax5.plot(
        sxx, np.log2(1 / f * result['dt']), 'gray', label='Fourier spectrum'
    )
    ax5.plot(
        result['global_ws'], np.log2(result['period']), 'b',
        label='Wavelet spectrum'
    )
    ax5.plot(
        result['global_signif'], np.log2(result['period']), 'r--',
        label='95% confidence spectrum'
    )
    ax5.legend(loc=0)
    ax5.set_xlim(0, 1.25 * np.max(result['global_ws']))
    ax5.set_xlabel('Power', fontsize=10)
    ax5.set_title('Global Wavelet Spectrum', fontsize=12)

    # save fig
    if not kwargs.get('no_plot', False):
        filename = kwargs.get('filename', '{}.png'.format(var))
        fig.savefig(filename, dpi=300)

    ret_dict = {
        'fig': fig,
        'ax_data': ax1,
        'ax_cwt': ax2,
        'ax_wavelet': ax3,
        # 'ax:': ax4,
        'ax_global_spectrum': ax5,
    }
    return ret_dict
