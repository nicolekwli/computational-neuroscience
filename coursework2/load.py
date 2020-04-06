from poisson import get_spike_count, get_spike_intervals, calc_fano_fac, calc_coeff_var
import matplotlib.pyplot as plt
import numpy as np

def load_data(filename,T):

    data_array = [T(line.strip()) for line in open(filename, 'r')]

    return data_array

# not sure if this is correct
def get_spike_count(spikes, windowSize):
    # get spikes for the given window size
    spikesInWindow = int(windowSize / 0.002)
    counts = []
    count = 0
    loopCount = 0
    for i in range (len(spikes)):
        loopCount += 1
        if (spikes[i] == 1):
            count += 1

        if (loopCount == spikesInWindow):
            counts.append(count)
            loopCount = 0
            count = 0
    return counts

def get_spike_times(spikes):
    intervals = []
    time = 0
    # Assuming the first is at 2ms
    for s in range (len(spikes)):
        if (spikes[s] == 1):
            time = (s+1) * 0.002
            intervals.append(time)

    return intervals

def autocorrelogram(spikes):
    '''
    occurance = [0] * 200
    for s in range (len(spikes)):
        for i in range (-100, 100):
            if (i != 0):
                if ((0 <= (s + i)) and ((s + i) < len(spikes))):
                    if (spikes[s+i] == 1):
                        occurance[i+100] += 1
    print(occurance)

    occurance[:] = [o / len(spikes) for o in occurance]
    '''
    inter = []
    for s in range (len(spikes)):
        if (spikes[s] == 1):
            time = ( s+1 ) * 0.002
            inter.append(time)
        else:
            inter.append(0)

    # print(occurance)
    # list(range(-200, 200)),
    plt.acorr(inter, maxlags = 100)
    plt.show()


def get_spike_trig_avg(spike_times, stimulus, window):
    loopCount = 0
    sta = [0] * 50
    for s in range (len(spike_times)):
        # Get list of times to get from stimulus
        windowStart = int(spike_times[s] * 1000 - 50)
        windowEnd = int(spike_times[s] * 1000 + 50)
        if (windowStart < 0):
            windowStart = 0
        if (windowEnd > (20 * 60 * 1000)):
            windowEnd = 20*60*1000

        timings = np.arange(windowStart, windowEnd, 2, dtype = int)
        for i in range (len(timings)):
            # add the stimules calue into bin
            # timings is in non decimal
            stiPos = int((timings[i-1] / 2) - 1)
            sta[i-1] += stimulus[stiPos]
    sta = [x / len(spike_times) for x in sta]
    return sta

def show_trig_avg_plot(xs, ys):
    plt.bar(xs, ys)
    plt.show()

# Q2---------------------------------------------
print("------------------START------------------")
#spikes=[int(x) for x in load_data("rho.dat")]
spikes=load_data("rho.dat",int)

# print(len(spikes))
# print(spikes[35:40])

# need to get spike_train for the window
window = 10 * 0.001
# countList = get_spike_count(spikes, window)
# calc_fano_fac(countList)

intervalsList = get_spike_times(spikes)
i_list = get_spike_intervals(intervalsList)
# calc_coeff_var(i_list)

# Q3------------------------------------------------------
# Plot autocorrelogram over the range -200ms - 200ms
# autocorrelogram(spikes)


# Q4------------------------------------------------------
#stimulus=[float(x) for x in load_data("stim.dat")]
stimulus=load_data("stim.dat",float)
window = 100 * 0.001

trigger = get_spike_trig_avg(intervalsList, stimulus, window)
print(len(trigger))
print(len(spikes))
print(len(stimulus))
xs = np.arange(-50,50,2)
show_trig_avg_plot(xs,trigger)
# print(len(stimulus))
# print(stimulus[0:5])

