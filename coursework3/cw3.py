import numpy as np
import matplotlib.pyplot as plt
import math
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


sec = 1.0
ms = 0.001
mV = 0.001   # millivolt
nA = 0.000000001
ohms = 0.001



# QUESTION A1----------------------------
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
    plt.xlabel('Time (msec)')
    plt.ylabel('Membrane Potential (V)')
    plt.show()



# QUESTION A2----------------------------

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
    plotVoltage(times, V)


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
    


      