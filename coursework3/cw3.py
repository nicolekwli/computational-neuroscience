import numpy as np
import matplotlib.pyplot as plt
import math
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
        
        # Use Euler's method with timestep delta_t = 0.25ms
        # Assume no refractory period
        #Ans1: Plot voltage as a function of time


# Cm: total membrane capacitance
# V = membrane potential
# tau_m = RmCm
'''
Cm(dv/dt) = ( E_L - V )/ m_rests + I_e : volt (unless refractory)
'''

Hz = 1.0
sec = 1.0
ms = 0.001
mV = 0.001   # millivolt
nA = 0.000000001
ohms = 0.001



# QUESTION A1---------------------------------------------------------
# V(t + dt) = V(T) + dt(f(V,t))
def getVoltageForTimes():
    V = np.zeros(len(times))

    for i in range (len(times)):
        if (i == 0):
            V[i] = 0
        else:
            # v(i-1) + dv/dt * dt
            V[i] = V[i-1] + ((E_L - V[i-1] + (m_resistance * I_e)) / m_tau) * dt
   
        if (V[i] >= v_threshold):
            V[i] = v_rest

    return V

def plotVoltage(times, V):
    plt.plot(times, V)

    plt.title('Integrate-and-Fire')
    plt.xlabel('Time (msec)')
    plt.ylabel('Membrane Potential (V)')
    plt.show()

def A1():
    pass


# QUESTION A2---------------------------------------------------
def simulateTwoNeurons():
    S_12 = np.zeros(len(times))
    S_21 = np.zeros(len(times))

    V_1 = np.zeros(len(times))
    V_1[0] = (random.randint(int(v_rest / mV), int(v_threshold /mV))) * mV
    V_2 = np.zeros(len(times))
    V_2[0] = (random.randint(int(v_rest / mV), int(v_threshold /mV))) * mV

    for i in range (len(times)):
        if (i == 0):
            S_12[i] = 0
            S_21[i] = 0

        else:
            # need to update s for n1 then use that to get current 
            current = getSynapseCurrent(V_1[i-1], S_21[i])
            V_1[i] = getVoltage(V_1[i-1], current)

            if (V_1[i] >= v_threshold):
                V_1[i] = v_rest
                S_12[i] = S_12[i-1] + P
                # print(V_1[i])

            else :
                S_12[i] = updateSynapse(S_12[i-1])

            
            current = getSynapseCurrent(V_2[i], S_12[i])
            V_2[i] = getVoltage(V_2[i-1], current)

            if (V_2[i] >= v_threshold):
                V_2[i] = v_rest
                S_21[i] = S_21[i-1] + P
            else:
                S_21[i] = updateSynapse(S_21[i-1])

    return V_1, V_2

def getVoltage(volt, sypCur):
    nextVolt = volt + (E_L - volt + Rm_Ie + sypCur) / m_tau * dt
    return nextVolt

def updateSynapse(s):
    return (s + dt * (-s/s_tau))

# This is Is(t) = gs * s(t) * (Es - V)
def getSynapseCurrent(volt, s):
    # g_bar_x * s(t) * (Es - V)
    Is_t = Rm_Gs * s * (E_s - volt)
    return Is_t

def plotVoltage2(times, V, V1):
    plt.plot(times, V)
    plt.plot(times, V1)

    plt.title('Simulation of two neurons')
    plt.xlabel('Time (msec)')
    plt.ylabel('Membrane Potential (V)')
    plt.show()


# QUESTION B1---------------------------------------------------
def genSpikeTrain():
    spikes = np.zeros(40)
    for s in range (len(spikes)):
        spike = random.uniform(0.00, 1.00)
        if (spike < r*dt):
            spikes[s] = 1
        else:
            spikes[s] = 0
    return spikes

# for each time
def simulateNeuron(ss, gs, volt):
    current = np.zeros(40)
    for c in range (40):
        current[c] =  R_m * gs[c] * ss[c] * (E_s - volt)
    # do everything but for all 40 synapses at the same tie

    # get one votlage
    nextVolt = volt + (E_L - volt + sum(current))/ m_tau * dt

    # check if spike 
    if (nextVolt >= v_threshold):
        nextVolt = v_rest
        #_12[i] = S_12[i-1] + P

    else :
        pass
        # S_12[i] = updateSynapse(S_12[i-1])

    return nextVolt


# MAIN ----------------------------------------------------------
if __name__ == "__main__":
    print("-----PartA Question1-----")
    
    m_tau = 10 * ms

    E_L = -70 * mV
    v_rest = -70 * mV
    v_threshold = -40 * mV
    m_resistance = 10 * ohms # m_omega

    I_e = 3.1  # nA

    duration = 1 * sec
    dt = 0.25 * ms
    times = np.arange(0, duration + dt, dt)

    refrac = 0

    V = getVoltageForTimes()
    # plotVoltage(times, V)

    # ----------------------------------------------------------------------------------------------------------------------
    print("-----PARTA QUESTION2-----")
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

    V_1, V_2 = simulateTwoNeurons()
    # plotVoltage2(times, V_1, V_2)

    # Inhibitory 
    E_s = -80 * mV

    V_1, V_2 = simulateTwoNeurons()
    # plotVoltage2(times, V_1, V_2)

    # ----------------------------------------------------------------------------------------------------------------------
    print("-----PARTB QUESTION1-----")

    # neuron params
    E_L = -65 * mV
    v_rest = -65 * mV
    v_threshold = -50 * mV
    v_reset = -65 * mV

    # passive membrane leak conductance 
    R_m = 100 * ohms
    m_tau = 10 * ms

    # do not include input current
    I_e = 0

    # 40 incoming synapses, all conductance based
        # single-exponential timecourse
        # use ODE to solve 40 S and 1 Volt
    s_tau = 2 * ms  # decay time constant
    gbar_i = 4     # initial peak conductance nanaSeimens (strength)
    E_s = 0
    ds = 0.5

    dt = 0.25 * ms

    # input spike train = independent homogeneous poisson process
        # with same average firing rate <r>, fixed in time but spikes arrive randomly
        # getting the rest of the r
            # draw random number on the unit interval from each synapse at each timestep
            # if < r dt ->> spike has occured
            # r dt needs to be < 1
    r = 15  * Hz
    
    # print(spike_train)

    # initialise ss and gs
    s_i = np.zeros(40)
    g_is = np.zeros(40)
    #g_is.fill(g_i)

    V = np.zeros(len(times))
    V[0] = 0
    # update each synapse with each voltage, it should affect the same neuron 
    # (i.e R_m * I_s_1 + R_m * I_s_2 + ... + R_m * I_s_40)
    for i in range (1,len(times)):
        # 40 spikes for 40 synapses
        spike_train = genSpikeTrain()

        for s in range (len(s_i)):
            current = np.zeros(40)
            if (spike_train[s] == 1):
                s_i[s] = s_i[s] + P

            else :
                s_i[s] = updateSynapse(s_i[s])

            current[s] =  R_m * g_is[s] * s_i[s] * (E_s - V[i-1])
            # do everything but for all 40 synapses at the same tie

            # get one votlage
        V[i] = V[i-1] + (E_L - V[i-1] + sum(current))/ m_tau * dt
        if (V[i] >= v_threshold):
            V[i] = v_rest

        # Get Voltage using synapses
        # V[i] = simulateNeuron(s_i, g_is, V[i-1])

    
    plotVoltage(times, V)








      