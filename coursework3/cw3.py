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
def A1():
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
        plt.xlabel('Time (sec)')
        plt.ylabel('Membrane Potential (V)')
        # plt.show()
        plt.savefig('A1')

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
        #plt.show()
        plt.legend(loc="upper left")
        plt.savefig('A2 Inhib')

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

    #V_1, V_2 = simulateTwoNeurons()
    #plotVoltage2(times, V_1, V_2)

    # Inhibitory 
    E_s = -80 * mV

    V_1, V_2 = simulateTwoNeurons()
    plotVoltage2(times, V_1, V_2)


# QUESTION B1---------------------------------------------------
def B1():
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
            #spike = random.random()
            
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

    def getVoltagesB1():
        fire_rate_30 = 0
        # update each synapse with each voltage, it should affect the same neuron 
        for i in range (len(times)):
            # 40 spikes for 40 synapses
            spike_train = genSpikeTrain()
            current = 0
            if (i != 0):
                for s in range (1,len(s_i)):
                    if (spike_train[s] == 1):
                        s_i[s] = s_i[s-1] + ds

                    else :
                        s_i[s] = s_i[s-1] + ((-s_i[s-1] / s_tau) * dt)
                    current = current + gbar_i*s_i[s]
            # current[s] =  R_m * current * (E_s - V[i-1]) # THIS IS QUESTIONABLE
            current = R_m * current * (E_s - V[i-1])
            # V[i] = V[i-1] + (E_L - V[i-1] + R_m + current) * dt / m_tau
            V[i] = V[i-1] + (E_L - V[i-1] + current) * dt / m_tau
            if (V[i] >= v_threshold):
                V[i] = v_rest
                if (times[i] >= 270.0):
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

    # do not include input current
    I_e = 0

    s_tau = 2 * ms  # decay time constant
    gbar_i = 4     # initial peak conductance nanaSeimens (strength)
    E_s = 0
    ds = 0.5

    dt = 0.25 * ms

    r = 15  * Hz

    # initialise ss and gs
    s_i = np.zeros(40)
    g_is = np.zeros(40)
    # g_is.fill(gbar_i)

    V = np.zeros(len(times))
    V[0] = 0
    
    a = getVoltagesB1()
    plotVoltage(times, V)



# QUESTION B2 ON---------------------------------------------------
def B2_on():
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
            #spike = random.random()
            
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
        loop_count = 0 
        fire_count = 0
        fire_rate_30 = 0
        for i in range (len(times)):
            spike_train = genSpikeTrain()
            current = 0
            if (i != 0):
                for s in range (1,len(s_i)):
                    if (spike_train[s] == 1):
                        s_i[s] = s_i[s-1] + ds
                        if (s_i[s] < 0):
                            s_i[s] = 0
                        if (s_i[s] > 4):
                            s_i[s] = 4
                        pre_syn_spike_t[s] = times[i]
                        pre_post_diff = post_syn_spike_t - pre_syn_spike_t[s]
                        g_is[s] = g_is[s] + updateGi(pre_post_diff)

                    else :
                        s_i[s] = s_i[s-1] + ((-s_i[s-1] / s_tau) * dt)
                        if (s_i[s] < 0):
                            s_i[s] = 0
                        if (s_i[s] > 4):
                            s_i[s] = 4

                    current = current + g_is[s]*s_i[s]
        
            current = R_m * current * (E_s - V[i-1])
            V[i] = V[i-1] + (E_L - V[i-1] + current) * dt / m_tau
            if (V[i] >= v_threshold):
                V[i] = v_rest
                fire_count += 1
                post_syn_spike_t = times[i]
                
                for j in range (40):
                    pre_post_diff = post_syn_spike_t - pre_syn_spike_t[j]
                    g_is[j] = g_is[j] + updateGi(pre_post_diff)
                if (times[i] >= 270.0):
                    fire_rate_30 += 1

            loop_count += 1
            if (loop_count == (10*sec/dt)):
                fire_rate.append(fire_count)
                loop_count = 0
                fire_count = 0
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

    pre_post_diff = post_syn_spike_t - pre_syn_spike_t

    a_plus = 0.2
    a_min = 0.25
    tau_plus = 20 * ms
    tau_min = 20 * ms

    stdp = True 
    gbar_i = 4
    g_is = np.zeros(40)
    g_is.fill(gbar_i)

    s_i = np.zeros(40)

    r = 15  * Hz

    duration = 300 * sec
    dt = 0.25 * ms
    times = np.arange(0, duration + dt, dt)

    V = np.zeros(len(times))
    V[0] = 0

    fire_rate = []
    loop_count = 0
    fire_count = 0
    fire_rate_30 = getVoltages40Syn_On()
    plotVoltage(times, V)

    return g_is

# QUESTION B2 OFF---------------------------------------------------
def B2_Off_with_On_Results(g_is):
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
            #spike = random.random()
            
            if (spike < r*dt):
                spikes[s] = 1
            else:
                spikes[s] = 0
        return spikes

    # THIS IS CURRENTLY FOR OFF
    def getVoltages40Syn_Off():
        fire_rate_30 = 0
        # update each synapse with each voltage, it should affect the same neuron 
        for i in range (len(times)):
            # 40 spikes for 40 synapses
            spike_train = genSpikeTrain()
            current = 0
            if (i != 0):
                for s in range (1,len(s_i)):
                    if (spike_train[s] == 1):
                        s_i[s] = s_i[s-1] + ds
                        if (s_i[s] < 0):
                            s_i[s] = 0
                        if (s_i[s] > 4):
                            s_i[s] = 4

                    else :
                        s_i[s] = s_i[s-1] + ((-s_i[s-1] / s_tau) * dt)
                        if (s_i[s] < 0):
                            s_i[s] = 0
                        if (s_i[s] > 4):
                            s_i[s] = 4
                    current = current + gbar_i*s_i[s]
            # current[s] =  R_m * current * (E_s - V[i-1]) # THIS IS QUESTIONABLE
            current = R_m * current * (E_s - V[i-1])
            # V[i] = V[i-1] + (E_L - V[i-1] + R_m + current) * dt / m_tau
            V[i] = V[i-1] + (E_L - V[i-1] + current) * dt / m_tau
            if (V[i] >= v_threshold):
                V[i] = v_rest
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

    s_tau = 2 * ms  # decay time constant
    gbar_i = 4     # initial peak conductance nanaSeimens (strength)
    E_s = 0
    ds = 0.5

    pre_syn_spike_t = np.zeros(40)
    post_syn_spike_t = -1000

    pre_post_diff = post_syn_spike_t - pre_syn_spike_t

    a_plus = 0.2
    a_min = 0.25
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
    V[0] = 0

    fire_rate = []
    loop_count = 0
    fire_count = 0
    fire_rate_30 = 0 

    fire_rate_30 = getVoltages40Syn_Off()
    plotVoltage(times, V)

# QUESTION B3 ON---------------------------------------------------
def B3_On_10():
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
            #spike = random.random()
            
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
        loop_count = 0 
        fire_count = 0
        fire_rate_30 = 0
        for i in range (len(times)):
            spike_train = genSpikeTrain()
            current = 0
            if (i != 0):
                for s in range (1,len(s_i)):
                    if (spike_train[s] == 1):
                        s_i[s] = s_i[s-1] + ds
                        if (s_i[s] < 0):
                            s_i[s] = 0
                        if (s_i[s] > 4):
                            s_i[s] = 4
                        pre_syn_spike_t[s] = times[i]
                        pre_post_diff = post_syn_spike_t - pre_syn_spike_t[s]
                        g_is[s] = g_is[s] + updateGi(pre_post_diff)

                    else :
                        s_i[s] = s_i[s-1] + ((-s_i[s-1] / s_tau) * dt)
                        if (s_i[s] < 0):
                            s_i[s] = 0
                        if (s_i[s] > 4):
                            s_i[s] = 4

                    current = current + g_is[s]*s_i[s]
        
            current = R_m * current * (E_s - V[i-1])
            V[i] = V[i-1] + (E_L - V[i-1] + current) * dt / m_tau
            if (V[i] >= v_threshold):
                V[i] = v_rest
                fire_count += 1
                post_syn_spike_t = times[i]
                
                for j in range (40):
                    pre_post_diff = post_syn_spike_t - pre_syn_spike_t[j]
                    g_is[j] = g_is[j] + updateGi(pre_post_diff)
                if (times[i] >= 270.0):
                    fire_rate_30 += 1

            loop_count += 1
            if (loop_count == (10*sec/dt)):
                fire_rate.append(fire_count)
                loop_count = 0
                fire_count = 0
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

    r = 10  * Hz

    gbar_i = 4
    g_is = np.zeros(40)
    # g_is.fill(gbar_i)

    s_i = np.zeros(40)

    duration = 300 * sec
    dt = 0.25 * ms
    times = np.arange(0, duration + dt, dt)

    V = np.zeros(len(times))
    V[0] = 0
    fire_rate = []
    loop_count = 0
    fire_count = 0


    fire_rate_30 = getVoltages40Syn_On()
    plotVoltage(times, V)

# MAIN ----------------------------------------------------------
if __name__ == "__main__":
    print("-----PartA Question1-----")
    # A1()

    # ----------------------------------------------------------------------------------------------------------------------
    print("-----PARTA QUESTION2-----")
    A2()

    # ----------------------------------------------------------------------------------------------------------------------
    print("-----PARTB QUESTION1-----")
    # B1()

    # ----------------------------------------------------------------------------------------------------------------------
    print("-----PARTB QUESTION2 STDP ON-----")
    #g_is_from_on = B2_on()
    
    print("-----PARTB QUESTION2 HISTOGRAM-----")
    #plt.hist(g_is)
    #plt.show()

    print("-----PARTB QUESTION2 AVERAGE FIRING RATE OF POST NEURON-----")
    #fire_rate[:] = [f / len(fire_rate) for f in fire_rate]

    # x = np.arange(0, duration, 10)
    # plt.plot(x, fire_rate)
    # plt.show()
    print("-----PARTB QUESTION2 FIRING RATE OF POST NEURON 30 Seconds-----")

    print("-----PARTB QUESTION2 STDP OFF-----")
    #g_avg = np.mean(g_is_from_on)
    #B2_Off_with_On_Results(g_avg)

    
    print("-----PARTB QUESTION2 FIRING RATE OF POST NEURON OFF 30 Seconds-----")
    # print(fire_rate_30)


    print("-----PARTB QUESTION3 ON 10Hz-----")
    #B3_On_10()







      