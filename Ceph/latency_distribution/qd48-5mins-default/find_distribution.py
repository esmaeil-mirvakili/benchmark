import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels as sm
import matplotlib
import matplotlib.pyplot as plt
from pandas import read_csv
from dateutil import parser
from datetime import datetime, timedelta, timezone
import pytz
import signal


matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')


def signal_handler(signum, frame):
    raise Exception("Timed out!")
signal.signal(signal.SIGALRM, signal_handler)
# Create models from data
def best_fit_distribution(data, bins=200, ax=None, title='res'):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [
        st.alpha, st.anglit, st.arcsine, st.beta, st.betaprime, st.bradford, st.burr, st.cauchy, st.chi, st.chi2,
        st.cosine, st.dgamma, st.dweibull, st.erlang, st.expon, st.exponnorm, st.exponweib, st.exponpow, st.f,
        st.fatiguelife, st.fisk, st.foldcauchy, st.foldnorm, st.frechet_r, st.frechet_l, st.genlogistic, st.genpareto,
        st.gennorm, st.genexpon, st.genextreme, st.gausshyper, st.gamma, st.gengamma, st.genhalflogistic, st.gilbrat,
        st.gompertz, st.gumbel_r, st.gumbel_l, st.halfcauchy, st.halflogistic, st.halfnorm, st.halfgennorm,
        st.hypsecant, st.invgamma, st.invgauss, st.invweibull, st.johnsonsb, st.johnsonsu, st.ksone, st.kstwobign,
        st.laplace, st.levy, st.levy_l, st.levy_stable, st.logistic, st.loggamma, st.loglaplace, st.lognorm, st.lomax,
        st.maxwell, st.mielke, st.nakagami, st.ncx2, st.ncf, st.nct, st.norm, st.pareto, st.pearson3, st.powerlaw,
        st.powerlognorm, st.powernorm, st.rdist, st.reciprocal, st.rayleigh, st.rice, st.recipinvgauss, st.semicircular,
        st.t, st.triang, st.truncexpon, st.truncnorm, st.tukeylambda, st.uniform, st.vonmises, st.vonmises_line,
        st.wald, st.weibull_min, st.weibull_max, st.wrapcauchy
    ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    fits = {}
    num = len(DISTRIBUTIONS)
    i = 1
    for distribution in DISTRIBUTIONS:
        print(distribution.name + ' '+str(i/num)+'%')
        i += 1
        # if i > 3:
        #     break
        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data

                signal.alarm(5*60)  # Ten seconds
                try:
                    params = distribution.fit(data)
                except Exception as msg:
                    print("Timed out!")
                    continue

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    # end
                except Exception:
                    pass

                param_names = (distribution.shapes + ', loc, scale').split(', ') if distribution.shapes else ['loc', 'scale']
                param_str = ', '.join(['{}={:0.2f}'.format(k, v) for k, v in zip(param_names, params)])
                dist_str = '{}({})'.format(distribution.name, param_str)
                dist_data = {'name': distribution.name, 'sse': sse, 'params': dist_str}
                fits[distribution.name] = dist_data
                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass
    f = open(title+".txt", "w")
    for fit in (sorted(fits.values(), key=lambda x: x['sse'])):
        f.write(fit['name']+':\n')
        f.write('\tsse:' + str(fit['sse'])+'\n')
        f.write('\tparams:' + str(fit['params'])+'\n')
    f.close()
    return (best_distribution.name, best_params)

def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get sane start and end points of distribution
    start = dist.ppf(0.01, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.01, loc=loc, scale=scale)
    end = dist.ppf(0.99, *arg, loc=loc, scale=scale) if arg else dist.ppf(0.99, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf

def load_data(name):
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    data = np.load(name+'.npz')
    # restore np.load for future normal usage
    np.load = np_load_old
    return data['arr_0'].tolist()


x_bs_lat = load_data("x_bs_lat")
y_bs_lat = load_data("y_bs_lat")
if2 = 'flush_job_timestamps.csv'  # Compaction for L0
if3 = 'compact_job_timestamps.csv'  # Compaction for other levels
id2 = read_csv(if2, parse_dates=True, squeeze=True, sep=',', header=None)
id3 = read_csv(if3, parse_dates=True, squeeze=True, sep=',', header=None)

id2len = len(id2.values)
id3len = len(id3.values)

x2_compact = []  # flush(L0) timestamps
x3_compact = []  # compact(>= L1) timestamps
y2_compact = []  # dummp y value
y3_compact = []  # dummp y value
w2_compact = []  # durations(width of compaction)
w3_compact = []  # durations(width of compaction)
utc = pytz.UTC
for i in range(id2len):
    x2_compact.append((parser.parse(id2.values[i, 1]) - timedelta(hours=5)).replace(tzinfo=utc))
    w2_compact.append(id2.values[i, 5] / 1000000)
    y2_compact.append(0.05)
for i in range(id3len):
    x3_compact.append((parser.parse(id3.values[i, 1]) - timedelta(hours=5)).replace(tzinfo=utc))
    w3_compact.append(id3.values[i, 5] / 1000000)
    y3_compact.append(0.05)

# plt.plot(x2_compact, y2_compact, label='4096 KiB', marker='^', c='g', linestyle='')
# plt.plot(x3_compact, y3_compact, label='4096 KiB', marker='d', c='r', linestyle='')
#
# plt.plot(x_bs_lat, y_bs_lat, label='bluestore')
# # ax.set(xlabel='time stamps', ylabel='latency [secs]', title='BlueStore Latency Time Series')
# # plt.legend()
# plt.show()
# plt.close()


def find_best_dist(data, title):
    # Load data from statsmodels datasets
    # data = pd.Series(sm.datasets.elnino.load_pandas().data.set_index('YEAR').values.ravel())
    data = pd.Series(data)

    # Plot for comparison
    plt.figure(figsize=(12,8))
    ax = data.plot(kind='hist', bins=50, density=True, alpha=0.5, color='b')
    # Save plot limits
    dataYLim = ax.get_ylim()

    # Find best fit distribution
    best_fit_name, best_fit_params = best_fit_distribution(data, 1_000_000, ax, title=title)
    print(best_fit_name)
    print(best_fit_params)
    best_dist = getattr(st, best_fit_name)

    # Update plots
    # ax.set_ylim(dataYLim)
    ax.set_title(title)
    ax.set_xlabel(u'Latency')
    ax.set_ylabel('Frequency')
    plt.savefig(title + ' all.png')
    # Make PDF with best params
    pdf = make_pdf(best_dist, best_fit_params)

    # Display
    plt.figure(figsize=(12,8))
    ax = pdf.plot(lw=2, label='PDF', legend=True)
    data.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Data', legend=True, ax=ax)

    param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
    dist_str = '{}({})'.format(best_fit_name, param_str)

    ax.set_title(title + ' best dist: ' + dist_str)
    ax.set_xlabel(u'Latency')
    ax.set_ylabel('Frequency')
    # plt.show()
    plt.savefig(title + ' best.png')

def number_of_writes_before(write_times, marked_times):
    number_of_writes = len(write_times)
    number_of_writes_before = []
    write_time_index = 0
    for marked_time in marked_times:
        count = 0
        while write_time_index < number_of_writes and write_times[write_time_index] < marked_time:
            write_time_index += 1
            count += 1
        number_of_writes_before.append(count)
    return number_of_writes_before


find_best_dist(y_bs_lat, 'writes')
spikes = [lat for lat in y_bs_lat if lat >= 0.05]
find_best_dist(spikes, 'spikes')
find_best_dist(np.log10(y_bs_lat), 'writes Log10')
find_best_dist(np.log10(spikes), 'spikes Log10')
spike_times = [x_bs_lat[i] for i in range(len(x_bs_lat)) if y_bs_lat[i] >= 0.05]
number_of_writes_before_spike = number_of_writes_before(x_bs_lat, spike_times)
number_of_writes_before_spike_all = [t for t in number_of_writes_before_spike if t > 0]
find_best_dist(number_of_writes_before_spike_all, 'writes before spikes')
number_of_writes_before_spike = [t for t in number_of_writes_before_spike if t > 10]
find_best_dist(number_of_writes_before_spike, 'writes before spikes more than 10')
number_of_writes_before_compaction = number_of_writes_before(x_bs_lat, x3_compact)
number_of_writes_before_compaction = [t for t in number_of_writes_before_compaction if t > 0]
find_best_dist(number_of_writes_before_compaction, 'writes before compaction')
number_of_writes_before_compaction = [t for t in number_of_writes_before_compaction if t > 10]
find_best_dist(number_of_writes_before_compaction, 'writes before compaction more than 10')
number_of_writes_before_flush = number_of_writes_before(x_bs_lat, x2_compact)
number_of_writes_before_flush = [t for t in number_of_writes_before_flush if t > 0]
find_best_dist(number_of_writes_before_flush, 'writes before flush')
number_of_writes_before_flush = [t for t in number_of_writes_before_flush if t > 10]
find_best_dist(number_of_writes_before_flush, 'writes before flush more than 10')
