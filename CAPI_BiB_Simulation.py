################################################
# CAPI - Crescent Active Particle Interactions #
################################################
#Author: Ludwig A. Hoffmann

################################################
# Balls-into-bins simulations.
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


def Pick_Random_Box(N):
    List_Boxes = np.random.randint(0,L,size=N)
    
    return List_Boxes

for m in range(len(Array_Number_Particles)):
    
    Initial_Number_Particles = Array_Number_Particles[m]
    
    for Run_Iteration in range(Number_Runs):
        
        f = open("Output_" + str(Initial_Number_Particles) + "/Run_" + str(Run_Iteration) + ".csv", 'w')
        writer = csv.writer(f)
    
        N_Cluster = np.zeros(Drawings)
        Number_Free_Particles = Initial_Number_Particles
        count_arr = np.zeros(L,dtype=int)
        
        for i in range(Drawings):
            
            Occupation_Boxes = np.zeros(L,dtype=int)
        
            List_Boxes = Pick_Random_Box(int(Number_Free_Particles))
            
            for j in range(Number_Free_Particles):
                Occupation_Boxes[int(List_Boxes[j])] += 1
            
            count_arr += Occupation_Boxes
            
            if i > 0:
                
                N_Cluster[i] = N_Cluster[i-1]
            
            for j in range(len(Occupation_Boxes)):
                
                if count_arr[j] > 1:
                    
                    N_Cluster[i] += Occupation_Boxes[j]
                        
                    Number_Free_Particles -= Occupation_Boxes[j]
                    
                if count_arr[j] < 2:
                    
                    count_arr[j] = 0
        
        writer.writerow(N_Cluster)