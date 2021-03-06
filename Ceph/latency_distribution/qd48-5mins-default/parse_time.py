import sys
import numpy as np
from pandas import read_csv
import pandas as pd
from dateutil import parser
from datetime import datetime, timedelta, timezone
import pytz
import matplotlib.pyplot as plt

utc = pytz.UTC  # work with offset-naive and offset-aware datetimes

f1 = 'dump_time_stamps_vec.csv'
data1 = read_csv(f1, skiprows=10000, parse_dates=True, squeeze=True, sep=',', header=None)
datalen = len(data1.values)


def check_bracket(str):
    if str == None:
        return "0"
    if str.startswith('['):
        return str[1:]
    if str.endswith(']'):
        return str[:-1]
    return str


# for bluestore latency
x_bs_lat = []
y_bs_lat = []
print("len = " + str(datalen))
# process the time stamps
# for i in range(datalen - 1):
#     # simple writes
#     # if len(data1.values[i,:]) == len(data1.values[i+1,:]) and data1.values[i,2] == 'simple_s':
#     if data1.values[i, 2] == 'simple_s':
#         # for first ctx
#         ctr_ctx1 = parser.parse(check_bracket(data1.values[i, 1])).replace(tzinfo=utc)
#         simple_s1 = parser.parse(check_bracket(data1.values[i, 3])).replace(tzinfo=utc)
#         aio_done1 = parser.parse(check_bracket(data1.values[i, 5])).replace(tzinfo=utc)
#         flush_cmt_s1 = parser.parse(check_bracket(data1.values[i, 7])).replace(tzinfo=utc)
#         flush_cmt_e1 = parser.parse(check_bracket(data1.values[i, 9])).replace(tzinfo=utc)
#         simple_e1 = parser.parse(check_bracket(data1.values[i, 11])).replace(tzinfo=utc)
#         # for second ctx
#         '''ctr_ctx2 = parser.parse(check_bracket(data1.values[i+1,1]))
#         simple_s2 = parser.parse(check_bracket(data1.values[i+1,3]))
#         aio_done2 = parser.parse(check_bracket(data1.values[i+1,5]))
#         flush_cmt_s2 = parser.parse(check_bracket(data1.values[i+1,7]))
#         flush_cmt_e2 = parser.parse(check_bracket(data1.values[i+1,9]))
#         simple_e2 = parser.parse(check_bracket(data1.values[i+1,11]))'''
#         # sanity check of timestamps
#         if simple_s1 < ctr_ctx1 or aio_done1 < simple_s1 or flush_cmt_s1 < aio_done1 or flush_cmt_e1 < flush_cmt_s1 or simple_e1 < flush_cmt_e1:
#             print("simple writes timestamp order is incorrect")
#         # bluestore latency
#         bluestore_lat_simple = simple_e1 - simple_s1
#         x_bs_lat.append(simple_s1)
#         y_bs_lat.append(bluestore_lat_simple.total_seconds())
#
#         # spikes
#         if bluestore_lat_simple.total_seconds() > 0.05:
#             print(str(i/datalen)+'%')
#             print("bluestore_lat_simple", bluestore_lat_simple.total_seconds(), ", simple_start",
#                   check_bracket(data1.values[i, 3]))
#
#     # deferred writes
#     elif data1.values[i, 2] == 'deferred_s':
#         ctr_ctx1 = parser.parse(check_bracket(data1.values[i, 1])).replace(tzinfo=utc)
#         deferred_s1 = parser.parse(check_bracket(data1.values[i, 3])).replace(tzinfo=utc)
#         flush_cmt_s1 = parser.parse(check_bracket(data1.values[i, 5])).replace(tzinfo=utc)
#         flush_cmt_e1 = parser.parse(check_bracket(data1.values[i, 7])).replace(tzinfo=utc)
#         deferred_e1 = parser.parse(check_bracket(data1.values[i, 9])).replace(tzinfo=utc)
#         # sanity check of timestamps
#         if deferred_s1 < ctr_ctx1 or flush_cmt_s1 < deferred_s1 or flush_cmt_e1 < flush_cmt_s1 or deferred_e1 < flush_cmt_e1:
#             print("deferred writes timestamp order is incorrect")
#         # bluestore latency
#         bluestore_lat_deferred = deferred_e1 - deferred_s1
#         x_bs_lat.append(deferred_s1)
#         y_bs_lat.append(bluestore_lat_deferred.total_seconds())
#
#         # spikes
#         if bluestore_lat_deferred.total_seconds() > 0.05:
#             print("bluestore_lat_deferred", bluestore_lat_deferred.total_seconds(), ", deferred_start",
#                   check_bracket(data1.values[i, 3]))


def store(data_list, name):
    data = np.array(data_list)
    np.savez(name, data)


def load_data(name):
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    data = np.load(name+'.npz')
    # restore np.load for future normal usage
    np.load = np_load_old
    return data['arr_0'].tolist()


# store(x_bs_lat, "x_bs_lat")
# store(y_bs_lat, "y_bs_lat")
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

for i in range(id2len):
    x2_compact.append((parser.parse(id2.values[i, 1]) - timedelta(hours=5)).replace(tzinfo=utc))
    w2_compact.append(id2.values[i, 5] / 1000000)
    y2_compact.append(0.05)
for i in range(id3len):
    x3_compact.append((parser.parse(id3.values[i, 1]) - timedelta(hours=5)).replace(tzinfo=utc))
    w3_compact.append(id3.values[i, 5] / 1000000)
    y3_compact.append(0.05)

plt.plot(x2_compact, y2_compact, label='4096 KiB', marker='^', c='g', linestyle='')
plt.plot(x3_compact, y3_compact, label='4096 KiB', marker='d', c='r', linestyle='')

plt.plot(x_bs_lat, y_bs_lat, label='bluestore')
# ax.set(xlabel='time stamps', ylabel='latency [secs]', title='BlueStore Latency Time Series')
# plt.legend()
plt.show()
plt.close()


# CDF
def plot_cdf(data, title, x_label):
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.xscale('log')
    n, bins, patches = ax.hist(
        data,
        1_000_000,
        density=True,
        histtype='step',
        cumulative=True,
        label='cdf'
    )
    ax.grid(True)
    ax.legend(loc='right')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Likelihood of occurrence')
    plt.show()
    plt.close()

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


plot_cdf(y_bs_lat, 'write latency cdf', 'latency')

spikes = [lat for lat in y_bs_lat if lat >= 0.05]
plot_cdf(spikes, 'spikes latency cdf', 'latency')

spike_times = [x_bs_lat[i] for i in range(len(x_bs_lat)) if y_bs_lat[i] >= 0.05]
number_of_writes_before_spike = number_of_writes_before(x_bs_lat, spike_times)
number_of_writes_before_spike = [n for n in number_of_writes_before_spike if n > 10]
plot_cdf(number_of_writes_before_spike, '# of writes before spike cdf', '# of writes')

number_of_writes_before_compaction = number_of_writes_before(x_bs_lat, x3_compact)
number_of_writes_before_compaction = [n for n in number_of_writes_before_compaction if n > 10]
plot_cdf(number_of_writes_before_compaction, '# of writes before compaction cdf', '# of writes')

number_of_writes_before_flush = number_of_writes_before(x_bs_lat, x2_compact)
number_of_writes_before_flush = [n for n in number_of_writes_before_flush if n > 10]
plot_cdf(number_of_writes_before_flush, '# of writes before flush cdf', '# of writes')
