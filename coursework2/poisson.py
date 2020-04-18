import random as rnd
import math as m

# Generates intervals according to the distribution
# Define SI units = 1.0
# Input:
#   rate = overall firing rate
#   tau_ref = refractory period
# Variable:
#   exp_rate = Poisson process muct have to calc correct overall
#              firing rate fiven refractory period
# Return: Spike trains for the interval [0, big_t]
def get_spike_train(rate, big_t, tau_ref):

    if 1 <= rate * tau_ref:
        print("firing rate not possible given refractory period f/p")
        return []

    exp_rate = rate / (1 - tau_ref * rate)

    spike_train = []

    t = rnd.expovariate(exp_rate)

    while t < big_t:
        spike_train.append(t)
        t += tau_ref + rnd.expovariate(exp_rate)

    return spike_train


# This returns a list of spike counts for individual windows
def get_spike_count(train, bigT, windowWidth):
    count = 0
    counts = []

    windowNum = int(bigT / windowWidth)

    # get the spike count for each window
    for i in range (0, windowNum):
        for spike in train:
            if ((spike >= i*windowWidth) and (spike < (i+1)*windowWidth)):
                count += 1
        counts.append(count)
        count = 0
    return counts

# Calculate Fano factor
# Applied to spike count

# Divide spike train into intervals
# Find the spike count for each interval
# THEN get variance and avg of these counts
#   F = delta^2 / miu
#   F = variance / avg
def calc_fano_fac(counts):
    mean = sum(counts) / len(counts)
    var = 0
    for c in counts:
        var += (c - mean) ** 2
    var = var / len(counts)
    # sd = m.sqrt(var)
    print("FANO: ", str(var/mean))
    # coeff = calc_coeff_var(m.sqrt(var), mean)
    # print("COEFF: ", str(m.sqrt(var)/mean))
    return (var/mean)


# get time difference between spikes
def get_spike_intervals(train):
    intervals = []
    intervals.append(train[0])
    for i in range (1, len(train)):
        intervals.append(train[i] - train[i-1])
    return intervals

# coefficient of variation = SD / avg --> applied to inter-spike interval = time difference between successive spikes
def calc_coeff_var(intervals):
    mean = sum(intervals) / len(intervals)
    var = 0
    for i in intervals:
        var += (i - mean) ** 2
    var = var / len(intervals)
    sd = m.sqrt(var)
    coeff = sd / mean
    print("COEFF: ", coeff)
    return coeff


if __name__ == "__main__":
    Hz = 1.0
    sec = 1.0
    ms = 0.001

    rate = 35.0 * Hz
    big_t = 1000 * sec

    tau_ref = 0 * ms
    window_width = 10 * ms

    spike_train = get_spike_train(rate, big_t, tau_ref)

    #print(len(spike_train))
    #print(len(spike_train) / big_t)
    # print(spike_train)

    # for refract = 0, windowwidth 10ms
    print("----- refracotry = 0, windowwidth = 10ms")
    s_counts = get_spike_count(spike_train, big_t, window_width)
    calc_fano_fac(s_counts)
    s_intervals = get_spike_intervals(spike_train)
    calc_coeff_var(s_intervals)

    # window 50ms
    print("----- refracotry = 0, windowwidth = 50ms")
    window_width = 50 * ms
    s_counts = get_spike_count(spike_train, big_t, window_width)
    calc_fano_fac(s_counts)
    s_intervals = get_spike_intervals(spike_train)
    calc_coeff_var(s_intervals)


    # window 100ms
    print("----- refracotry = 0, windowwidth = 100ms")
    window_width = 100 * ms
    s_counts = get_spike_count(spike_train, big_t, window_width)
    calc_fano_fac(s_counts)
    s_intervals = get_spike_intervals(spike_train)
    calc_coeff_var(s_intervals)

    print("----- refracotry = 5ms, windowwidth = 10ms")
    tau_ref = 5 * ms
    spike_train = get_spike_train(rate, big_t, tau_ref)

    window_width = 10 * ms

    s_counts = get_spike_count(spike_train, big_t, window_width)
    calc_fano_fac(s_counts)
    s_intervals = get_spike_intervals(spike_train)
    calc_coeff_var(s_intervals)


    print("----- refracotry = 5ms, windowwidth = 50ms")
    window_width = 50 * ms

    s_counts = get_spike_count(spike_train, big_t, window_width)
    calc_fano_fac(s_counts)
    s_intervals = get_spike_intervals(spike_train)
    calc_coeff_var(s_intervals)

    print("----- refracotry = 5ms, windowwidth = 100ms")
    window_width = 100 * ms

    s_counts = get_spike_count(spike_train, big_t, window_width)
    calc_fano_fac(s_counts)
    s_intervals = get_spike_intervals(spike_train)
    calc_coeff_var(s_intervals)