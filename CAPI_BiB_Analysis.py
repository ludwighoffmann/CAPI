################################################
# CAPI - Crescent Active Particle Interactions #
################################################
#Author: Ludwig A. Hoffmann

################################################
# Analysis of balls-into-bins simulations.
################################################

import numpy as np
import math
import random
import csv

#Fix model parameters
BoxSize = 5
L = int((1200/BoxSize)**2)
iterations = 130000
v0 =  0.04
Drawings = int(iterations * v0/BoxSize)
Array_Number_Particles = [10,30,60]
Number_Runs = 50


for m in range(len(Array_Number_Particles)):
    
    Initial_Number_Particles = Array_Number_Particles[m]
    N_Cluster_Average = np.zeros(Drawings)
    csv_import = np.zeros((Drawings,Number_Runs))
        
    for Run_Iteration in range(Number_Runs):
          
            csv_import[:,Run_Iteration] = np.genfromtxt("Output_" + str(Initial_Number_Particles) + "/Run_" + str(Run_Iteration) + ".csv", delimiter=',')    
    
    for j in range(Drawings):
        N_Cluster_Average[j] = csv_import[j,:].sum()/Number_Runs
        
        
    xpoints = np.linspace(1,iterations*0.16/60,Drawings)
    
    ypoints = N_Cluster_Average/Initial_Number_Particles
    
    plt.plot(xpoints,ypoints)
        
plt.show()