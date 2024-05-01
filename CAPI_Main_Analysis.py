################################################
# CAPI - Crescent Active Particle Interactions #
################################################
#Author: Ludwig A. Hoffmann


################################################
# Analysis of simulation data for different densities and angles.
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

N_particle = 60 #Number of particles, i.e., the density

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

array_number_particles = [10,20,30,40,50] #List of all N_particle we ran simulations for. 



for Numb in array_number_particles:
    
    """
    Function that for each density computes the number of clustered particles in each run at each time step.
    """
    
    array_number_particles_cluster_all_runs = []
    number_of_clusters_of_give_size_all_runs = np.zeros((Total_Number_Runs,Reduced_length_csv-2,10),dtype=int)
    
    for Run_Iteration in range(Total_Number_Runs):
      
        csv_import = np.genfromtxt("Output_"+ str(Numb) + "/Position_" + str(Numb) + "_" + str(Radius_Particles) + "_Run_" + str(Run_Iteration) + ".csv", delimiter=',')
         
        reshape = csv_import.reshape(Length_csv,Numb,2)
        reduced_array = np.array(reshape[0])
        
        for i in range(1,Length_csv): #Reduce length of array by Reduction_factor_csv
            if i%Reduction_factor_csv == 0:
                reduced_array = np.vstack((reduced_array,reshape[i]))
        
        reduced_array = reduced_array.reshape(Reduced_length_csv,Numb,2) #reshape reduced array into correct format
        
        result = np.diff(reduced_array,axis = 0) #take difference between consecutive time steps by subtracting neighboring columns
        
        accounting_particles_in_cluster = np.zeros((Reduced_length_csv-2,Numb))
        
        for t in range(Reduced_length_csv-2): #count particle as being in a cluster if it moves a distance less than v0*Reduction_factor_csv*10/1.5 in the Reduction_factor_csv*10 timesteps that are inbetween two neighboring columns in reduced_array 
            for n in range(Numb):
                dx = min(abs(result[t,n,0]), abs(L-abs(result[t,n,0])))
                dy = min(abs(result[t,n,1]), abs(L-abs(result[t,n,1])))
                dist = np.sqrt(dx**2 + dy**2)
                if dist < v0*Reduction_factor_csv*10/1.5: #1.5:
                    accounting_particles_in_cluster[t,n] = 1
        number_part_in_cluster = np.sum(accounting_particles_in_cluster,axis = 1) #total number of particle in cluster is sum over array accounting_particles_in_cluster    
        array_number_particles_cluster_all_runs.append(number_part_in_cluster) #append total number of particles in cluster to array counting the total number of all runs    
        
        number_of_particles_in_cluster = np.zeros((Reduced_length_csv-2,Numb),dtype=int)
        number_of_clusters_of_give_size = np.zeros((Reduced_length_csv-2,10),dtype=int) #8 here means we are looking for clusters of at most 7 particles, bigger clusters would be missed by this, can be found by increasing number here
        
        for t in range(Reduced_length_csv-2): #count how many particles in a given array: for every particle that is in an array count the particles that are closer than 3 * Radius_Particles as part of the same array
            for n in range(Numb):
                if accounting_particles_in_cluster[t,n] == 1:
                    for m in range(Numb):
                        if np.sqrt((reduced_array[t,n,0]-reduced_array[t,m,0])**2+(reduced_array[t,n,1]-reduced_array[t,m,1])**2) < 3 * Radius_Particles:
                            number_of_particles_in_cluster[t,n] += 1
            number_of_clusters_of_give_size[t,:] = np.bincount(number_of_particles_in_cluster[t,:],None,10)
        
        number_of_clusters_of_give_size_all_runs[Run_Iteration,:,:] =  number_of_clusters_of_give_size[:,:] #array storing values for all arrays
    
    average_number_particles_clusters = np.array(array_number_particles_cluster_all_runs).sum(axis=0)/Total_Number_Runs #average over all runs how many particles in cluster at given time
    
    time_evolution_percent_certain_cluster_size = np.zeros((Reduced_length_csv-2,10))
    
    for t in range(Reduced_length_csv-2): #time evolution of average of how many clusters of certain particle size
        time_evolution_percent_certain_cluster_size[t] = number_of_clusters_of_give_size_all_runs[:,t,:].sum(axis = 0)/Total_Number_Runs/Numb*100
    
    
    average_number_particles_clusters.tofile("180AngleOutput_"+ str(Numb) + "/average_number_particles_clusters.csv", sep = ',') #store data for plot with plot.py
    time_evolution_percent_certain_cluster_size.tofile("180AngleOutput_"+ str(Numb) + "/time_evolution_percent_certain_cluster_size.csv", sep = ',')