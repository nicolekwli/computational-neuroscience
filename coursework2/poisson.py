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


def get_spike_count(spike_train):
    count = 0
    counts = []

    # get the spike count for wach window
    for i in range (0, int(big_t/window_width)):
        for spike in spike_train:
            if ((spike >= i*window_width) and (spike < i*window_width+window_width)):
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
    coeff = calc_coeff_var(m.sqrt(var), mean)
    print("COEFF: ", str(coeff))
    return (var/mean)

# coefficient of variation = SD / avg --> applied to inter-spike interval = time difference between successive spikes
# ISI is 1000
# am confused
def calc_coeff_var(sd, avg):
    return (sd/avg)


Hz = 1.0
sec = 1.0
ms = 0.001

# original 15
rate = 35.0 * Hz
# original 5
tau_ref = 0 * ms

# original 5 * sec
# I assume this is the ISI
big_t = 1 * sec
window_width = 100 * ms

spike_train = get_spike_train(rate, big_t, tau_ref)

print("Spike train for the interval " + str(big_t))
print(len(spike_train) / big_t)

print("Spike Train")
print(spike_train)

s_counts = get_spike_count(spike_train)
calc_fano_fac(s_counts)
