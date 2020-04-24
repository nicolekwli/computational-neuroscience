from poisson import get_spike_count, get_spike_intervals, calc_fano_fac, calc_coeff_var
import matplotlib.pyplot as plt
import numpy as np

def load_data(filename,T):

    data_array = [T(line.strip()) for line in open(filename, 'r')]

    return data_array

# not sure if this is correct
def get_spike_count_for_rho(spikes, windowSize):
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
    # Assuming the first is at 2ms which would make 20mins have sense
    for s in range (len(spikes)):
        if (spikes[s] == 1):
            time = (s+1) * 0.002
            intervals.append(time)

    return intervals

def autocorrelogram(spikes):

    occurance = [0] * 100
    for s in range (len(spikes)):
        if (spikes[s] == 1):
            for i in range (-50, 50):
                #if (i != 0):
                try:
                    if (spikes[s+i] == 1):
                        occurance[i+50] += 1
                except:
                    pass

    occurance[:] = [o / len(spikes) for o in occurance]

    x = np.arange(-100,100,2)

    #plt.plot(x, occurance)
    #plt.title('Autocorrelogram')
    #plt.xlabel('Time (ms)')
    #plt.ylabel('Membrane Potential (V)')
    #plt.show()
    #plt.savefig('q3_2.png')



def get_spike_trig_avg(spike_times, stimulus, window):
    sta = [0] * 50
    for s in range (len(spike_times)):
        # Get list of times to get from stimulus
        # Convert to int
        windowStart = int(spike_times[s] * 1000 - 100)
        windowEnd = int(spike_times[s] * 1000)
        if (windowStart < 0):
            windowStart = 0

        timings = np.arange(windowStart, windowEnd, 2, dtype = int)
        for i in range (len(timings)):
            # add the stimules calue into bin
            # timings is in non decimal

            # Stipos = stimulus value at that timing
            stiPos = int((timings[i-1] / 2) - 1)
            sta[i-1] += stimulus[stiPos]
    sta = [x / len(spike_times) for x in sta]
    return sta

def show_trig_avg_plot(xs, ys):
    plt.plot(xs, ys)
    # plt.show()
    plt.title('Spike Triggered Average')
    plt.xlabel('Time (ms)')
    plt.savefig('q4_2.png')

# Q2---------------------------------------------
print("------------------START------------------")
print("RHO DAT 20 minutes Sampling rate 500Hz")
#spikes=[int(x) for x in load_data("rho.dat")]
spikes=load_data("rho.dat",int)

# need to get spike_train for the window
print("window: 10ms")
window = 10 * 0.001
countList = get_spike_count_for_rho(spikes, window)
calc_fano_fac(countList)

intervalsList = get_spike_times(spikes)
i_list = get_spike_intervals(intervalsList)
calc_coeff_var(i_list)

print("window: 50ms")
window = 50 * 0.001
countList = get_spike_count_for_rho(spikes, window)
calc_fano_fac(countList)

intervalsList = get_spike_times(spikes)
i_list = get_spike_intervals(intervalsList)
calc_coeff_var(i_list)

print("window: 100ms")
window = 100 * 0.001
countList = get_spike_count_for_rho(spikes, window)
calc_fano_fac(countList)

intervalsList = get_spike_times(spikes)
i_list = get_spike_intervals(intervalsList)
calc_coeff_var(i_list)

# Q3------------------------------------------------------
# Plot autocorrelogram over the range -200ms - 200ms
autocorrelogram(spikes)


# Q4------------------------------------------------------
#stimulus=[float(x) for x in load_data("stim.dat")]
stimulus=load_data("stim.dat",float)
window = 100 * 0.001

trigger = get_spike_trig_avg(intervalsList, stimulus, window)
print(len(trigger))

xs = np.arange(-100,0,2)
show_trig_avg_plot(xs,trigger)


