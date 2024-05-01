################################################
# CAPI - Crescent Active Particle Interactions #
################################################
#Author: Ludwig A. Hoffmann


################################################
# Analysis of simulation data for time-dependent concentration.
################################################

import numpy as np
import random
import csv
import os
import shutil
import multiprocessing as mp
from itertools import starmap, combinations
from operator import add

#Fix model parameters
L = 1137

N_particle = 1 #Actual number of particles will be determined below as this values changes over time.

N_segments = 9
N_Disks = N_particle * N_segments
v0 =  0.04
iterations = 300000 #Number of iterations -> total length
OpeningAngle = np.pi #Opening angle: np.pi = 180°, np.pi/2 = 90°, np.pi/3 = 60°, ...
Radius_Particles = np.pi/OpeningAngle

Length_csv = int(iterations/10) #We saved data every 10 time steps in the simulations.
Reduction_factor_csv = 10
Reduced_length_csv = int(Length_csv / Reduction_factor_csv)

Iteration_Counter = 0
Total_Number_Runs = 100


array_number_particles = [784] #list of all N_particle we ran simulations for
Array_FitPara_MaxPart = [[8.28,165,96],[18.63,405,133],[34.78,784,174]]
IndexFinalConcentration = 2

N_particle = Array_FitPara_MaxPart[IndexFinalConcentration][2]

def IntegerListForAdding(FitPara,MaxParticle,Offset):
    
    """
    Computes a list to be used to add particles to the system over time. Specifies at which iteration steps particles are added.
    """
    
    ListInteger = []
    
    for i in range(Offset+1,MaxParticle+1):
        Tmp_Int = math.floor(375 * (i - Offset)**2 / (FitPara**2))
    
        ListInteger.append(Tmp_Int)
    
    return(ListInteger)

def ListNumbParticles():
    
    N_particle = Array_FitPara_MaxPart[IndexFinalConcentration][2]
    Iteration_Counter = 0
    NumbParticlesList = []

    ListInteger = IntegerListForAdding(Array_FitPara_MaxPart[IndexFinalConcentration][0],Array_FitPara_MaxPart[IndexFinalConcentration][1],Array_FitPara_MaxPart[IndexFinalConcentration][2])

    for i in range(iterations):
        if(N_particle < Array_FitPara_MaxPart[IndexFinalConcentration][1]):
            if(Iteration_Counter == ListInteger[N_particle-Array_FitPara_MaxPart[IndexFinalConcentration][2]]):
                N_particle += 1
            if(Iteration_Counter%10==0):
                NumbParticlesList.append(N_particle)
        else:
            if(Iteration_Counter%10==0):
                NumbParticlesList.append(N_particle)
        
        Iteration_Counter += 1
        
    return(NumbParticlesList)


def reshape_func(Run_Iteration):
    
    """
    Reshape the imported csv files according to the particles in the system at each time step (found from Array_FitPara_MaxPart).
    """
    
    
    csv_import = np.genfromtxt("Output_"+ str(Array_FitPara_MaxPart[IndexFinalConcentration][1]) + "/Position_" + str(Array_FitPara_MaxPart[IndexFinalConcentration][2])  + "_" + str(Radius_Bananas) + "_Run_" + str(Run_Iteration) + ".csv", delimiter=',')
    
    reshape = np.zeros((Length_csv,Array_FitPara_MaxPart[IndexFinalConcentration][1],2))
    
    NumbParticlesList = ListNumbParticles()
    
    Pos_Counter = 0
    
    for t in range(Length_csv):
        
        reshape[t,:NumbParticlesList[t],:] = csv_import[Pos_Counter:Pos_Counter + NumbParticlesList[t],:]
        
        Pos_Counter += NumbParticlesList[t]
        
    return(reshape,NumbParticlesList)
    



for Numb in array_number_particles:
    
    """
    Function that for each density computes the number of clustered particles in each run at each time step.
    """
    
    array_number_particles_cluster_all_runs = []
    number_of_clusters_of_give_size_all_runs = np.zeros((Total_Number_Runs,Reduced_length_csv-2,40),dtype=int)
    
    for Run_Iteration in range(Total_Number_Runs):
        
        reshape,NumbParticlesList = reshape_func(Run_Iteration)
    
        reduced_array = np.zeros((Reduced_length_csv,Array_FitPara_MaxPart[IndexFinalConcentration][1],2))
        
        reduced_array[0,:,:] = reshape[0,:,:]
        
        j = 0
        
        for i in range(1,Length_csv): #Reduce length of array by Reduction_factor_csv
            
            if(i%Reduction_factor_csv == 0):
                
                j += 1
                reduced_array[j,:,:] = reshape[i,:,:]
        
        result = np.diff(reduced_array,axis = 0) #take difference between consecutive time steps by subtracting neighboring columns #Need to somehow take boundary conditions into account?!
        
        accounting_particles_in_cluster = np.zeros((Reduced_length_csv,Numb))
        
        for t in range(Reduced_length_csv-1): #count particle as in cluster if it moved a distance less than v0*Reduction_factor_csv*10/1.5 in the Reduction_factor_csv*10 timesteps that are inbetween two neighboring columns in reduced_array 
            for n in range(NumbParticlesList[t*10]):
                dx = min(abs(result[t,n,0]), abs(L-abs(result[t,n,0])))
                dy = min(abs(result[t,n,1]), abs(L-abs(result[t,n,1])))
                dist = np.sqrt(dx**2 + dy**2)
                if dist < v0*Reduction_factor_csv*10/1.5: #1.5:
                    accounting_particles_in_cluster[t,n] = 1
        
        number_part_in_cluster = np.sum(accounting_particles_in_cluster,axis = 1) #total number of particle in cluster is sum over array accounting_particles_in_cluster
        
        array_number_particles_cluster_all_runs.append(number_part_in_cluster) #append total number of particles in cluster to array counting the total number of all runs
        
        
        number_of_particles_in_cluster = np.zeros((Reduced_length_csv-2,Numb),dtype=int)
        number_of_clusters_of_give_size = np.zeros((Reduced_length_csv-2,40),dtype=int) #8 here means we are looking for clusters of at most 7 particles, bigger clusters would be missed by this, can be found by increasing number here
        
        for t in range(Reduced_length_csv-2): #count how many particles in a given array: for every particle that is in an array count the particles that are closer than 3 * Radius_Bananas as part of the same array
            for n in range(Numb):
                if accounting_particles_in_cluster[t,n] == 1:
                    for m in range(Numb):
                        if np.sqrt((reduced_array[t,n,0]-reduced_array[t,m,0])**2+(reduced_array[t,n,1]-reduced_array[t,m,1])**2) < 3 * Radius_Bananas:
                            number_of_particles_in_cluster[t,n] += 1
            number_of_clusters_of_give_size[t,:] = np.bincount(number_of_particles_in_cluster[t,:],None,40)
        
        number_of_clusters_of_give_size_all_runs[Run_Iteration,:,:] =  number_of_clusters_of_give_size[:,:] #array storing values for all arrays
    
    average_number_particles_clusters = np.array(array_number_particles_cluster_all_runs).sum(axis=0)/Total_Number_Runs #average over all runs how many particles in cluster at given time
    
    time_evolution_percent_certain_cluster_size = np.zeros((Reduced_length_csv-2,40))
    
    for t in range(Reduced_length_csv-2): #time evolution of average of how many clusters of certain particle size
        time_evolution_percent_certain_cluster_size[t] = number_of_clusters_of_give_size_all_runs[:,t,:].sum(axis = 0)/Total_Number_Runs/Numb*100
    
    
    average_number_particles_clusters.tofile("Output_"+ str(Numb) + "/average_number_particles_clusters.csv", sep = ',') #store data for plot with plot.py
    time_evolution_percent_certain_cluster_size.tofile("Output_"+ str(Numb) + "/time_evolution_percent_certain_cluster_size.csv", sep = ',')