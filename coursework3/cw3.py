import numpy as np
import matplotlib.pyplot as plt
import math as m
import random
# Part A 
    # Question 1
        # Simulate an integrate and fire model
        # Params for 1 second:
            # tau_m = 10ms              ----- Membrane time constant
            # E_L   
            # V_rest = -70 mV   ----- Leak potential, Reset Voltage
            # V_th  = -40mV             ----- Threshold
            # R_m   = 10 M_omega        ----- Membrane resistance (1/conductance)
            # I_e   = 3.1 nA            ----- input current?

# Cm: total membrane capacitance

Hz = 1.0
sec = 1.0
ms = 0.001
mV = 0.001   # millivolt

nA = 10**(-9)
ohms = 10**6
nS = 10**(-9)


# QUESTION A1---------------------------------------------------------
def A1():
    # V(t + dt) = V(T) + dt(f(V,t))
    def getVoltageForTimes():
        V = np.zeros(len(times))

        for i in range (len(times)):
            if (i == 0):
                V[i] = 0
            else:
                V[i] = V[i-1] + ((E_L - V[i-1] + (m_resistance * I_e)) / m_tau) * dt
    
            if (V[i] >= v_threshold):
                V[i] = v_rest

        return V

    def plotVoltage(times, V):
        plt.plot(times, V)

        plt.title('Integrate-and-Fire')
        plt.xlabel('Time (sec)')
        plt.ylabel('Membrane Potential (V)')
        # plt.show()
        plt.savefig('A1')

    m_tau = 10 * ms

    E_L = -70 * mV
    v_rest = -70 * mV
    v_threshold = -40 * mV
    m_resistance = 10 * ohms # m_omega

    I_e = 3.1 * nA # nA

    duration = 1 * sec
    dt = 0.25 * ms
    times = np.arange(0, duration + dt, dt)

    refrac = 0

    V = getVoltageForTimes()
    plotVoltage(times, V)


# QUESTION A2---------------------------------------------------
def A2():
    def simulateTwoNeurons():
        S_1 = np.zeros(len(times))
        S_2 = np.zeros(len(times))

        V_1 = np.zeros(len(times))
        V_1[0] = (random.randint(int(v_rest / mV), int(v_threshold /mV))) * mV
        V_2 = np.zeros(len(times))
        V_2[0] = (random.randint(int(v_rest / mV), int(v_threshold /mV))) * mV

        for i in range (len(times)):
            if (i == 0):
                S_1[i] = 0
                S_2[i] = 0

            else:
                if (V_1[i-1] == v_rest):
                    S_2[i] = S_2[i-1] + updateSynapse(S_2[i-1]) + P
                else:
                    S_2[i] = S_2[i-1] + updateSynapse(S_2[i-1])

                current = getSynapseCurrent(V_2[i-1], S_2[i])
                V_2[i] = V_2[i-1] + getVoltage(V_2[i-1], current)

                if (V_2[i] >= v_threshold):
                    V_2[i] = v_rest

                #-------------------------------------------------
                if (V_2[i-1] == v_rest):
                    S_1[i] = S_1[i-1] + updateSynapse(S_1[i-1]) + P
                else:
                    S_1[i] = S_1[i-1] + updateSynapse(S_1[i-1])

                current = getSynapseCurrent(V_1[i-1], S_1[i])
                V_1[i] = V_1[i-1] + getVoltage(V_1[i-1], current)

                if (V_1[i] >= v_threshold):
                    V_1[i] = v_rest


        return V_1, V_2

    def getVoltage(volt, sypCur):
        nextVolt = (E_L - volt + Rm_Ie + sypCur) / m_tau * dt
        return nextVolt

    def updateSynapse(s):
        return (dt * (-s/s_tau))

    def getSynapseCurrent(volt, s):
        Is_t = Rm_Gs * s * (E_s - volt)
        return Is_t

    def plotVoltage2(times, V, V1):
        plt.plot(times, V, color="b", label="neuron 1")
        plt.plot(times, V1, color="c", label="neuron 2")

        plt.title('Simulation of two neurons')
        plt.xlabel('Time (sec)')
        plt.ylabel('Membrane Potential (V)')
        
        plt.legend(loc="upper left")
        #plt.show()
        plt.savefig('A2 Inhit')

    duration = 1 * sec
    dt = 0.25 * ms
    times = np.arange(0, duration + dt, dt)

    # Neuron Params
    m_tau = 20 * ms

    E_L = -70 * mV
    v_rest = -80 * mV
    v_threshold = -54 * mV
    Rm_Ie = 18 * mV

    # Synapse Params
    Rm_Gs = 0.15
    P = 0.5
    s_tau = 10 * ms

    # Excitatory
    E_s = 0 * mV

    # V_1, V_2 = simulateTwoNeurons()
    # plotVoltage2(times, V_1, V_2)

    # Inhibitory 
    E_s = -80 * mV

    V_1, V_2 = simulateTwoNeurons()
    plotVoltage2(times, V_1, V_2)


# QUESTION B1---------------------------------------------------
def B1():
    def plotVoltage(times, V):
        plt.plot(times, V)

        plt.title('Integrate-and-Fire')
        plt.xlabel('Time (sec)')
        plt.ylabel('Membrane Potential (V)')
        # plt.show()
        plt.savefig('B1')

    def genSpikeTrain():
        spikes = np.zeros(40)
        for s in range (len(spikes)):
            spike = random.uniform(0.00, 1.00)
            if (spike < r*dt):
                spikes[s] = 1
            else:
                spikes[s] = 0
        return spikes

    def getVoltagesB1():
        fire_rate_30 = 0
        for i in range (len(times)):
            spike_train = genSpikeTrain()
            current = 0
            if (i != 0):
                for s in range (1,len(s_i)):
                    if (spike_train[s] == 1):
                        s_i[s] = s_i[s-1] + ((-s_i[s-1] / s_tau) * dt) + ds

                    else :
                        s_i[s] = s_i[s-1] + ((-s_i[s-1] / s_tau) * dt)
                    current = current + gbar_i*s_i[s]
            current = R_m * current * (E_s - V[i-1])
            V[i] = V[i-1] + (E_L - V[i-1] + current) * dt / m_tau
            if (V[i] >= v_threshold):
                V[i] = v_rest
                
                fire_rate_30 += 1

        return fire_rate_30

    duration = 1 * sec
    dt = 0.25 * ms
    times = np.arange(0, duration + dt, dt)

    # neuron params
    E_L = -65 * mV
    v_rest = -65 * mV
    v_threshold = -50 * mV
    v_reset = -65 * mV

    # passive membrane leak conductance 
    R_m = 100 * ohms
    m_tau = 10 * ms

    I_e = 0

    s_tau = 2 * ms  # decay time constant
    gbar_i = 4 * nS    # initial peak conductance nanaSeimens (strength)
    E_s = 0
    ds = 0.5

    dt = 0.25 * ms

    r = 15  * Hz

    # initialise ss and gs
    s_i = np.zeros(40)
    g_is = np.zeros(40)
    # g_is.fill(gbar_i)

    V = np.zeros(len(times))
    V[0] = v_rest
    
    a = getVoltagesB1()
    print(a)
    plotVoltage(times, V)

# QUESTION B2 ON---------------------------------------------------
def B2_on():
    def plotVoltage(times, V):
        plt.plot(times, V)
        plt.title('Integrate-and-Fire')
        plt.xlabel('Time (sec)')
        plt.ylabel('Membrane Potential (V)')
        plt.show()

    def plotSynWeightHist(g_is):
        plt.title('Steady-State Synaptic Weights Distribution')
        plt.hist(g_is, bins=10)

        plt.xlabel('Weights (nS)')
        plt.ylabel('Frequency')
        #plt.show()
        plt.savefig('sssslabelled4')

    def plotAvgFireRate300(fires):
        timesBins = np.arange(0, 300, 10)

        # divide by 10 seconds
        fires[:] = [f / 10 for f in fires]
        print(fires)

        plt.title('Average firing rate')

        plt.xlabel('Time (sec)')
        plt.ylabel('Firing Rate')
        plt.plot(timesBins, fires)
        
        # plt.show()
        plt.savefig('firerate300')

    def plotAvgFireRate30(fires):
        print(fires)
        timesBins = np.arange(270, 300, 1)

        fires[:] = [f / 30 for f in fires]
        print(np.mean(fires))

        plt.title('firing rate')
        plt.xlabel('Time (sec)')
        plt.ylabel('Firing Rate')
        plt.plot(timesBins, fires)
        
        #plt.show()
        plt.savefig('firerate30_on')

    def genSpikeTrain():
        spikes = np.zeros(40)
        for s in range (len(spikes)):
            spike = random.uniform(0.00, 1.00)
            if (spike < r*dt):
                spikes[s] = 1
            else:
                spikes[s] = 0
        return spikes

    def updateGi(pre_post_diff):
        delta_t = 0
        if (pre_post_diff > 0):
            delta_t = a_plus * m.exp(- (abs(pre_post_diff)) / tau_plus)
        else:
            delta_t = -a_min * m.exp(- (abs(pre_post_diff)) / tau_min)
        return delta_t

    def getVoltages40Syn_On():
        fire_index = 0
        fire_count = 0
        fire_rate_300 = np.zeros(30)
        
        fire_rate_30 = 0
        for i in range (len(times)):
            spike_train = genSpikeTrain()
            current = np.zeros(40)
            if (i != 0):
                for s in range (1,len(s_i)):
                    if (spike_train[s] == 1):
                        s_i[s] = s_i[s] + ((-s_i[s] / s_tau) * dt) + ds

                        pre_syn_spike_t[s] = times[i]
                        pre_post_diff = post_syn_spike_t - pre_syn_spike_t[s]
                        # depression
                        g_is[s] = g_is[s] + updateGi(pre_post_diff)
                        if (g_is[s] < 0):
                            g_is[s] = 0
                        if (g_is[s] > 4 * nS):
                            g_is[s] = 4 * nS

                    else :
                        s_i[s] = s_i[s] + ((-s_i[s] / s_tau) * dt)

                    current[s] = g_is[s]*s_i[s]
        
            V[i] = V[i-1] + (E_L - V[i-1] + np.sum(current) * R_m * (E_s - V[i-1])) * dt / m_tau
            if (V[i] >= v_threshold):
                V[i] = v_rest
                post_syn_spike_t = times[i]
                
                for j in range (40):
                    # potentiation
                    pre_post_diff = post_syn_spike_t - pre_syn_spike_t[j]
                    g_is[j] = g_is[j] + updateGi(pre_post_diff)

                    if (g_is[j] < 0):
                        g_is[j] = 0
                    if (g_is[j] > 4 * nS):
                        g_is[j] = 4 * nS

                fire_rate_300[fire_index] = fire_rate_300[fire_index] + 1.0

                if (times[i] >= 270.0):
                    fire_rate_30 = fire_rate_30 +  1

            fire_count += 1
            
            if (fire_count == (10.0/dt)):
                fire_count = 0
                fire_index += 1
        return fire_rate_300, fire_rate_30

    # neuron params
    E_L = -65 * mV
    v_rest = -65 * mV
    v_threshold = -50 * mV
    v_reset = -65 * mV

    # passive membrane leak conductance 
    R_m = 100 * ohms
    m_tau = 10 * ms

    s_tau = 2 * ms

    E_s = 0
    ds = 0.5

    pre_syn_spike_t = np.zeros(40)
    post_syn_spike_t = -1000

    pre_post_diff = post_syn_spike_t - pre_syn_spike_t

    a_plus = 0.2 * nS
    a_min = 0.25 * nS
    tau_plus = 20 * ms
    tau_min = 20 * ms

    # stdp = True 
    gbar_i = 4 * nS
    g_is = np.zeros(40)
    g_is.fill(gbar_i)

    s_i = np.zeros(40)

    r = 15  * Hz

    duration = 300 * sec
    dt = 0.25 * ms
    times = np.arange(0, duration + dt, dt)

    V = np.zeros(len(times))
    V[0] = v_rest

    fire_rate_300, fire_rate_30 = getVoltages40Syn_On()

    print("-----PARTB QUESTION2 HISTOGRAM-----")
    plotSynWeightHist(g_is)
    
    print("-----PARTB QUESTION2 firing rate 300-----")
    #plotAvgFireRate300(fire_rate_300)
    
    print(fire_rate_30)
    print(fire_rate_30/30)
    #plotAvgFireRate30(fire_rate_30)
    # plotVoltage(times, V)
    
    return g_is

# QUESTION B2 OFF---------------------------------------------------
def B2_Off_with_On_Results(g_avg):
    def plotVoltage(times, V):
        plt.plot(times, V)

        plt.title('Integrate-and-Fire')
        plt.xlabel('Time (msec)')
        plt.ylabel('Membrane Potential (V)')
        plt.show()

    def plotAvgFireRate30(fires):
        print(fires)
        timesBins = np.arange(270, 300, 1)


        fires[:] = [f / 30 for f in fires]
        print(np.mean(fires))

        plt.title('firing rate')

        plt.xlabel('Time (sec)')
        plt.ylabel('Firing Rate')
        plt.plot(timesBins, fires)
        
        #plt.show()
        #plt.savefig('firerate30_off_and_on_results2')

    def genSpikeTrain():
        spikes = np.zeros(40)
        for s in range (len(spikes)):
            spike = random.uniform(0.00, 1.00)
            if (spike < r*dt):
                spikes[s] = 1
            else:
                spikes[s] = 0
        return spikes

    # THIS IS CURRENTLY FOR OFF
    def getVoltages40Syn_Off():
        fire_rate_30 = 0

        for i in range (len(times)):
            spike_train = genSpikeTrain()
            current = np.zeros(40)
            if (i != 0):
                for s in range (1,len(s_i)):
                    if (spike_train[s] == 1):
                        s_i[s] = s_i[s] + ((-s_i[s] / s_tau) * dt) + ds

                    else :
                        s_i[s] = s_i[s] + ((-s_i[s] / s_tau) * dt)
                       
                    current[s] = g_is[s]*s_i[s]
            V[i] = V[i-1] + (E_L - V[i-1] + np.sum(current) * R_m * (E_s - V[i-1])) * dt / m_tau
            if (V[i] >= v_threshold):
                V[i] = v_rest
                if (times[i] >= 270.0):
                    fire_rate_30 = fire_rate_30 + 1
        return fire_rate_30

    # neuron params
    E_L = -65 * mV
    v_rest = -65 * mV
    v_threshold = -50 * mV
    v_reset = -65 * mV

    # passive membrane leak conductance 
    R_m = 100 * ohms
    m_tau = 10 * ms

    s_tau = 2 * ms  # decay time constant
    gbar_i = 4 * nS     # initial peak conductance nanaSeimens (strength)
    E_s = 0
    ds = 0.5

    pre_syn_spike_t = np.zeros(40)
    post_syn_spike_t = -1000

    pre_post_diff = post_syn_spike_t - pre_syn_spike_t

    a_plus = 0.2 * nS
    a_min = 0.25 * nS
    tau_plus = 20 * ms
    tau_min = 20 * ms

    g_is = np.zeros(40)
    g_is.fill(g_avg)

    s_i = np.zeros(40)

    r = 15  * Hz

    duration = 300 * sec
    dt = 0.25 * ms
    times = np.arange(0, duration + dt, dt)

    V = np.zeros(len(times))
    V[0] = v_rest

    fire_rate = []
    loop_count = 0
    fire_count = 0
    fire_rate_30 = 0 

    fire_rate_30 = getVoltages40Syn_Off()
    print(fire_rate_30)
    print(fire_rate_30/30)
    # print(np.mean(fire_rate_30))
    #plotAvgFireRate30(fire_rate_30)
    
    # plotVoltage(times, V)

# QUESTION B3 ON---------------------------------------------------
def B3_On_10(r):
    def plotVoltage(times, V):
        plt.plot(times, V)
        plt.title('Integrate-and-Fire')
        plt.xlabel('Time (msec)')
        plt.ylabel('Membrane Potential (V)')
        plt.show()

    def genSpikeTrain():
        spikes = np.zeros(40)
        for s in range (len(spikes)):
            spike = random.uniform(0.00, 1.00)

            if (spike < r*dt):
                spikes[s] = 1
            else:
                spikes[s] = 0
        return spikes

    def updateGi(pre_post_diff):
        delta_t = 0
        if (pre_post_diff > 0):
            delta_t = a_plus * m.exp(- (abs(pre_post_diff)) / tau_plus)
        else:
            delta_t = -a_min * m.exp(- (abs(pre_post_diff)) / tau_min)
        return delta_t
        
    def getVoltages40Syn_On():
        fire_rate_30 = 0
        for i in range (len(times)):
            spike_train = genSpikeTrain()
            current = 0
            if (i != 0):
                for s in range (1,len(s_i)):
                    if (spike_train[s] == 1):
                        s_i[s] = s_i[s] + ((-s_i[s] / s_tau) * dt) + ds
                       
                        pre_syn_spike_t[s] = times[i]
                        pre_post_diff = post_syn_spike_t - pre_syn_spike_t[s]
                        g_is[s] = g_is[s] + updateGi(pre_post_diff)
                        if (g_is[s] < 0):
                            g_is[s] = 0
                        if (g_is[s] > 4 * nS):
                            g_is[s] = 4 * nS

                    else :
                        s_i[s] = s_i[s] + ((-s_i[s] / s_tau) * dt)
                     

                    current = current + g_is[s]*s_i[s]
        
            current = R_m * current * (E_s - V[i-1])
            V[i] = V[i-1] + (E_L - V[i-1] + current) * dt / m_tau
            if (V[i] >= v_threshold):
                V[i] = v_rest
                post_syn_spike_t = times[i]
                
                for j in range (40):
                    pre_post_diff = post_syn_spike_t - pre_syn_spike_t[j]
                    g_is[j] = g_is[j] + updateGi(pre_post_diff)
                    if (g_is[j] < 0):
                        g_is[j] = 0
                    if (g_is[j] > 4 * nS):
                        g_is[j] = 4 * nS
                if (times[i] >= 270.0):
                    fire_rate_30 += 1
        return fire_rate_30

    # neuron params
    E_L = -65 * mV
    v_rest = -65 * mV
    v_threshold = -50 * mV
    v_reset = -65 * mV

    # passive membrane leak conductance 
    R_m = 100 * ohms
    m_tau = 10 * ms

    s_tau = 2 * ms

    E_s = 0
    ds = 0.5

    pre_syn_spike_t = np.zeros(40)
    post_syn_spike_t = -1000

    gbar_i = 4 * nS
    g_is = np.zeros(40)
    g_is.fill(gbar_i)

    a_plus = 0.2 * nS
    a_min = 0.25 * nS
    tau_plus = 20 * ms
    tau_min = 20 * ms

    s_i = np.zeros(40)

    duration = 300 * sec
    dt = 0.25 * ms
    times = np.arange(0, duration + dt, dt)

    V = np.zeros(len(times))
    V[0] = v_rest

    # r = 10  * Hz
    fire_rate_30 = getVoltages40Syn_On()
    print(fire_rate_30)
    print(fire_rate_30/30)
    rate = fire_rate_30
    return rate, g_is

# QUESTION B3 ON---------------------------------------------------
def B3_Off_10(r):
    def plotVoltage(times, V):
        plt.plot(times, V)

        plt.title('Integrate-and-Fire')
        plt.xlabel('Time (msec)')
        plt.ylabel('Membrane Potential (V)')
        plt.show()

    def plotAvgFireRate30(fires):
        print('fire rate for off')
        print(fires)
        timesBins = np.arange(270, 300, 1)

        fires[:] = [f / 30 for f in fires]
        print(np.mean(fires))

        plt.title('firing rate')

        plt.xlabel('Time (sec)')
        plt.ylabel('Firing Rate')
        plt.plot(timesBins, fires)
        
        #plt.show()
        #plt.savefig('firerate30_off_and_on_results2')

    def genSpikeTrain():
        spikes = np.zeros(40)
        for s in range (len(spikes)):
            spike = random.uniform(0.00, 1.00)
            if (spike < r*dt):
                spikes[s] = 1
            else:
                spikes[s] = 0
        return spikes

    def getVoltages40Syn_Off():
        fire_rate_30 = 0
        for i in range (len(times)):
            # 40 spikes for 40 synapses
            spike_train = genSpikeTrain()
            current = 0
            if (i != 0):
                for s in range (1,len(s_i)):
                    if (spike_train[s] == 1):
                        s_i[s] = s_i[s] + ((-s_i[s] / s_tau) * dt) + ds

                    else :
                        s_i[s] = s_i[s] + ((-s_i[s] / s_tau) * dt)
                     
                    current = current + g_is[s]*s_i[s]
            current = R_m * current * (E_s - V[i-1])
            V[i] = V[i-1] + (E_L - V[i-1] + current) * dt / m_tau
            if (V[i] >= v_threshold):
                V[i] = v_rest
                if (times[i] >= 270.0):
                    fire_rate_30 = fire_rate_30 + 1

        return fire_rate_30
    
    # neuron params
    E_L = -65 * mV
    v_rest = -65 * mV
    v_threshold = -50 * mV
    v_reset = -65 * mV

    # passive membrane leak conductance 
    R_m = 100 * ohms
    m_tau = 10 * ms

    s_tau = 2 * ms

    E_s = 0
    ds = 0.5

    pre_syn_spike_t = np.zeros(40)
    post_syn_spike_t = -1000

    gbar_i = 4 * nS
    g_is = np.zeros(40)
    g_is.fill(gbar_i)

    a_plus = 0.2 * nS
    a_min = 0.25 * nS
    tau_plus = 20 * ms
    tau_min = 20 * ms

    s_i = np.zeros(40)

    duration = 300 * sec
    dt = 0.25 * ms
    times = np.arange(0, duration + dt, dt)

    V = np.zeros(len(times))
    V[0] = v_rest

    fire_rate_30 = getVoltages40Syn_Off()
    print(fire_rate_30)
    print(fire_rate_30/30)
    rate = fire_rate_30
    return rate, g_is

def B4(B):
    def genSpikeTrain(t):
        spikes = np.zeros(40)
        value = r0 + (B * m.sin(2 * m.pi * f * t))
        for s in range (len(spikes)):
            spike = random.uniform(0.00, 1.00)

            if (spike < dt * value):
                spikes[s] = 1
            else:
                spikes[s] = 0
        return spikes

    def updateGi(pre_post_diff):
        delta_t = 0
        if (pre_post_diff > 0):
            delta_t = a_plus * m.exp(- (abs(pre_post_diff)) / tau_plus)
        else:
            delta_t = -a_min * m.exp(- (abs(pre_post_diff)) / tau_min)
        return delta_t

    def getVoltages40Syn_On():
        fire_rate_30 = 0
        for i in range (len(times)):
            spike_train = genSpikeTrain(times[i])
            current = 0
            if (i != 0):
                for s in range (1,len(s_i)):
                    if (spike_train[s] == 1):
                        s_i[s] = s_i[s] + ((-s_i[s] / s_tau) * dt) + ds
                     
                        pre_syn_spike_t[s] = times[i]
                        pre_post_diff = post_syn_spike_t - pre_syn_spike_t[s]
                        g_is[s] = g_is[s] + updateGi(pre_post_diff)
                        if (g_is[s] < 0):
                            g_is[s] = 0
                        if (g_is[s] > 4 * nS):
                            g_is[s] = 4 * nS

                    else :
                        s_i[s] = s_i[s] + ((-s_i[s] / s_tau) * dt)
                      

                    current = current + g_is[s]*s_i[s]
        
            current = R_m * current * (E_s - V[i-1])
            V[i] = V[i-1] + (E_L - V[i-1] + current) * dt / m_tau
            if (V[i] >= v_threshold):
                V[i] = v_rest
                post_syn_spike_t = times[i]
                
                for j in range (40):
                    pre_post_diff = post_syn_spike_t - pre_syn_spike_t[j]
                    g_is[j] = g_is[j] + updateGi(pre_post_diff)
                    if (g_is[j] < 0):
                        g_is[j] = 0
                    if (g_is[j] > 4 * nS):
                        g_is[j] = 4 * nS
                if (times[i] >= 270.0):
                    fire_rate_30 += 1
        return fire_rate_30

    def getVoltages40Syn_Off():
        fire_rate_30 = 0
        for i in range (len(times)):
            spike_train = genSpikeTrain()
            current = 0
            if (i != 0):
                for s in range (1,len(s_i)):
                    if (spike_train[s] == 1):
                        s_i[s] = s_i[s] + ((-s_i[s] / s_tau) * dt) + ds
                
                    else :
                        s_i[s] = s_i[s] + ((-s_i[s] / s_tau) * dt)
                    
                    current = current + g_is[s]*s_i[s]
            current = R_m * current * (E_s - V[i-1])
            V[i] = V[i-1] + (E_L - V[i-1] + current) * dt / m_tau
            if (V[i] >= v_threshold):
                V[i] = v_rest
                if (times[i] >= 270.0):
                    fire_rate_30 = fire_rate_30 + 1

        return fire_rate_30
    # neuron params
    E_L = -65 * mV
    v_rest = -65 * mV
    v_threshold = -50 * mV
    v_reset = -65 * mV

    # passive membrane leak conductance 
    R_m = 100 * ohms
    m_tau = 10 * ms

    s_tau = 2 * ms

    E_s = 0
    ds = 0.5

    pre_syn_spike_t = np.zeros(40)
    post_syn_spike_t = -1000

    gbar_i = 4 * nS
    g_is = np.zeros(40)
    g_is.fill(gbar_i)

    a_plus = 0.2 * nS
    a_min = 0.25 * nS
    tau_plus = 20 * ms
    tau_min = 20 * ms

    s_i = np.zeros(40)

    duration = 300 * sec
    dt = 0.25 * ms
    times = np.arange(0, duration + dt, dt)

    V = np.zeros(len(times))
    V[0] = v_rest

    r0 = 20  * Hz
    f = 10 * Hz

    # fire_rate_30 = getVoltages40Syn_Off()
    fire_rate_30 = getVoltages40Syn_On()
    print(B)
    print(fire_rate_30)
    print(fire_rate_30/30)

    return g_is

# MAIN ----------------------------------------------------------
if __name__ == "__main__":
    print("-----PartA Question1-----")
    #A1()

    # ----------------------------------------------------------------------------------------------------------------------
    print("-----PARTA QUESTION2-----")
    # A2()

    # ----------------------------------------------------------------------------------------------------------------------
    print("-----PARTB QUESTION1-----")
    #B1()

    # ----------------------------------------------------------------------------------------------------------------------
    print("-----PARTB QUESTION2 STDP ON-----")
    # g_is_from_on = B2_on()
    
    # print("-----PARTB QUESTION2 HISTOGRAM-----")
    # plt.hist(g_is)
    # plt.show()

    #print("-----PARTB QUESTION2 AVERAGE FIRING RATE OF POST NEURON-----")
    #fire_rate[:] = [f / len(fire_rate) for f in fire_rate]

    # x = np.arange(0, duration, 10)
    # plt.plot(x, fire_rate)
    # plt.show()
    #print("-----PARTB QUESTION2 FIRING RATE OF POST NEURON 30 Seconds-----")

    print("-----PARTB QUESTION2 STDP OFF-----")
    # g_avg = np.mean(g_is_from_on)
    # print(g_avg)
    # B2_Off_with_On_Results(g_avg)

    
    print("-----PARTB QUESTION2 FIRING RATE OF POST NEURON OFF 30 Seconds-----")
    # print(fire_rate_30)

    print("-----PARTB QUESTION3 ON 10-20Hz-----")
    rates = []

    # r = 10  * Hz
    # rate, gis = B3_On_10(r)
    # rates.append(rate)

    # r = 12  * Hz
    # rate, gis = B3_On_10(r)
    # rates.append(rate)

    # r = 14  * Hz
    # rate, gis = B3_On_10(r)
    # rates.append(rate)

    # r = 16  * Hz
    # rate, gis = B3_On_10(r)
    # rates.append(rate)

    # r = 18  * Hz
    # rate, gis = B3_On_10(r)
    # rates.append(rate)

    # r = 20  * Hz
    # rate, gis = B3_On_10(r)
    # rates.append(rate)
    # print(rates)

    # add the total of fires of 5 simulations
    # rates.append(47)
    # rates.append(32)
    # rates.append(13)
    # rates.append(9)
    # rates.append(8)
    # rates.append(7)

    # rates[:] = [f / 5 for f in rates]
    # rates[:] = [f / 30 for f in rates]

    # print(rates)

    print("-----PARTB QUESTION3 OFF 10-20Hz-----")

    rates_off = []

    # r = 10  * Hz
    # rate, gis = B3_Off_10(r)
    # rates_off.append(rate)

    # r = 12  * Hz
    # rate, gis = B3_Off_10(r)
    # rates_off.append(rate)

    # r = 14  * Hz
    # rate, gis = B3_Off_10(r)
    # rates_off.append(rate)

    # r = 16  * Hz
    # rate, gis = B3_Off_10(r)
    # rates_off.append(rate)

    # r = 18  * Hz
    # rate, gis = B3_Off_10(r)
    # rates_off.append(rate)

    # r = 20  * Hz
    # rate, gis = B3_Off_10(r)
    # rates_off.append(rate)
    # print(rates_off)

    # rates_off.append(362)
    # rates_off.append(1117)
    # rates_off.append(2310)
    # rates_off.append(3709)
    # rates_off.append(5480)
    # rates_off.append(7294)

    # rates_off[:] = [f / 5 for f in rates_off]
    # rates_off[:] = [f / 30 for f in rates_off]

    # x = np.arange(10,21,2)

    # plt.plot(x, rates, label="STDP ON")
    # plt.plot(x, rates_off, label="STDP OFF")

    # plt.title('Mean Output Firing Rate based on Input Firing Rate')
    # plt.xlabel('Input Firing Rate(Hz)')
    # plt.ylabel('Output Firing Rate(Hz)')

    # plt.legend(loc="upper left")
    # plt.savefig('B3_outputinput_onoff')

    print("-----PARTB QUESTION3 ssss distribution 10/20Hz-----")
    # r = 10  * Hz
    # rate, gis_10 = B3_On_10(r)

    # r = 20  * Hz
    # rate, gis_20 = B3_On_10(r)

    # plt.title('Steady-State Synaptic Weights')
    # plt.hist(gis_10, bins=10, fc=(0, 0, 1, 0.5), label="10Hz")
    # plt.hist(gis_20, bins=10, fc=(0, 1, 0.5, 0.5), label="20Hz")

    # plt.xlabel('Weights (nS)')
    # plt.ylabel('Frequency')
    # plt.legend(loc="upper left")
    # plt.savefig('B3 ssss1 label1')

    print("-----PARTB QUESTION4-----")
    gs_mean = []
    gs_std = []

    # B = 0
    # results = B4(B)
    # gs0 = np.mean(results)
    # gs_mean.append(gs0)

    # gs0 = np.std(results)
    # gs_std.append(gs0)

    # B = 5
    # results = B4(B)
    # gs5 = np.mean(results)
    # gs_mean.append(gs5)

    # gs5 = np.std(results)
    # gs_std.append(gs5)

    # B = 10
    # results = B4(B)
    # gs10 = np.mean(results)
    # gs_mean.append(gs10)

    # gs10 = np.std(results)
    # gs_std.append(gs10)

    # B = 15
    # results = B4(B)
    # gs15 = np.mean(results)
    # gs_mean.append(gs15)

    # gs15 = np.std(results)
    # gs_std.append(gs15)

    # B = 20
    # results = B4(B)
    # gs20 = np.mean(results)
    # gs_mean.append(gs20)

    # gs20 = np.std(results)
    # gs_std.append(gs20)

    # bs = [0, 5, 10, 15, 20]

    # #plt.title('Steady-State Synaptic Weights Mean')
    # plt.title('Steady-State Synaptic Weights Standard Deviation')
    # #plt.plot(bs, gs_mean, label="Mean")
    # plt.plot(bs, gs_std, label="Standard Deviation")

    # # plt.legend(loc="upper left")
    # # plt.xlabel('Degree of Correlation (Hz)')
    # # plt.ylabel('Mean (nS)')
    # plt.xlabel('Degree of Correlation (Hz)')
    # plt.ylabel('Standard Deviation')
    # plt.savefig('B4 std labels3')

    print("-----PARTB QUESTION4 DISTRIBUTION-----")

    # B = 0
    # gs0 = B4(B)

    # plt.title('Steady-State Synaptic Weights histogram for B=0Hz')
    # plt.hist(gs0, density=True, bins=10, fc=(0, 0, 1, 0.5))
    # plt.xlabel('Weight (nS)')
    # plt.ylabel('Frequency')
    # plt.savefig('B4 ssss_0hz label3')

    B = 20
    gs20 = B4(B)

    plt.title('Steady-State Synaptic Weights for B=20Hz')
    plt.hist(gs20, bins=10, fc=(0, 0, 1, 0.5))
    plt.xlabel('Weight (nS)')
    plt.ylabel('Frequency')
    plt.savefig('B4 ssss_20hz label6')


    # plt.title('Steady-State Synaptic Weights')
    # plt.hist(gs0, density=True, bins=10, fc=(0, 0, 1, 0.5), label="0Hz")
    # # plt.hist(gs20, density=True, bins=10, fc=(1, 0, 0, 0.5), label="10Hz")
    # plt.hist(gs20, density=True, bins=10, fc=(0, 1, 0, 0.5), label="20Hz")
    # plt.legend(loc="upper left")
    # plt.savefig('B4 ssss_020_2')


     





      