################################################
# CAPI - Crescent Active Particle Interactions #
################################################
#Author: Ludwig A. Hoffmann


################################################
# Main simulation used. By varying N_particle and OpeningAngle below we run the simulations for different densities and opening angles, respectively.
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
eta1 = 0.
eta2 = 0.

Total_Number_Runs = 100
Iteration_Counter = 0


def initilization():
    """
    Initialization function. First N_particle number of points are randomly distributed in space and their position is stored in the list pos_particle. For each of these we choose a random theta (orientation of the particle) and then add N_segments-1 number of points to each one, with position given lying on a circular segment with radius Radius_Particles. The position of these particles is then added to pos_particle (taking the orientation of the crescent particle into account by applying a rotation matrix containing theta to the point position) such that the first N_particle entries of pos_particle are the N_particle different crescent particles and after that each N_segments-1 block belongs to another crescent particle. Finally the orientation of the crescent particle is stored for each of the segment disks. 
    """
    
    pos_center_particle = np.random.uniform(0,L,size=(N_particle,2))
    pos_particle = []
    pos_particle.append(pos_center_particle)
    pos_particle = pos_particle[0].tolist()
    orient_particle_center = []

    for i in range(N_particle):

        theta = np.random.uniform(-np.pi, np.pi)

        angle_for_points = []
        Step_Angle = OpeningAngle/(N_segments-1)
        Angle = OpeningAngle/(N_segments-1)
        

        while Angle < OpeningAngle/2 + 0.01:
            angle_for_points.append([Radius_Particles * np.cos(3 * np.pi/2 + Angle),Radius_Particles * (np.sin(3 * np.pi/2 + Angle)+1)])
            angle_for_points.append([Radius_Particles * np.cos(3 * np.pi/2 - Angle),Radius_Particles * (np.sin(3 * np.pi/2 - Angle)+1)])
            Angle += Step_Angle
        
        
        Center_Of_Mass_Factor = (np.array(angle_for_points).sum(axis=0)/N_segments)[1]

        for j in range(len(angle_for_points)):
            
            pos_particle.append([pos_particle[i][0] + angle_for_points[j][0] * np.cos(2 * theta) - angle_for_points[j][1] * np.sin(2 * theta),pos_particle[i][1] + angle_for_points[j][1] * np.cos(2 * theta) + angle_for_points[j][0] * np.sin(2 * theta)])

        orient_particle_center.append(2 * theta + np.pi/2)

    orient_particle = []
    orient_particle = orient_particle_center
    for i in range(N_particle):
        for j in range(N_segments - 1):
            orient_particle.append(orient_particle[i])
    
    pos_particle = np.array(pos_particle)
    
    return(pos_particle,orient_particle,Center_Of_Mass_Factor)
def magnitude_angle_segments_array():
    
    """
    Computes the center of mass and the position of the segments relative to the center of mass and from this the angle and magnitude of the vector pointing from the com to the segment disk.
    """
    
    body_frame_array = np.zeros((N_Disks,2))
    angle_abs_com_segment_vec_array = np.zeros((N_Disks,2))

            
    pos_com = np.zeros((N_Disks,2))   
    for i in range(N_particle):
        com_x = pos_particle[i][0] + Center_Of_Mass_Factor * np.cos(orient_particle[i])
        com_y = pos_particle[i][1] + Center_Of_Mass_Factor * np.sin(orient_particle[i])
        pos_com[i] = [com_x,com_y]
    for j in range(N_particle):
        pos_com[N_particle + j * (N_segments-1) : N_particle + (j + 1) * (N_segments-1)] = np.stack([pos_com[j]]*(N_segments-1))
    
    body_frame_array = pos_particle - pos_com
    
    magnitude = np.sqrt((body_frame_array*body_frame_array).sum(axis=1))
    angle = np.real(np.arccos((np.stack((np.cos(orient_particle),np.sin(orient_particle)),axis=-1)*body_frame_array).sum(axis=1)/magnitude + 0j))
    angle[N_particle::2] *= -1
    angle_abs_com_segment_vec_array = np.stack((angle,magnitude),axis=-1)
            
    return(angle_abs_com_segment_vec_array)
def func_pair_potentials(a,b,pos_particle,orient_particle):
    
    """
    Compute the pair potential between two crescent particles. First, create an array with the position and orientation of all segements of the two crescent particles a and b and compute the difference between the two. To take the boundary conditions into account we look at the minimum between |a-b| and L-|a-b|. Every time we choose the second we need to reverse the orienation of the vector and that is what is done when going from Diff_2 to Diff. Then compute the grad and the angular derivative of the pair potential
    """
    
    u0 = 1
    FallOff = 1
    Lambda = 0.6
    
    grad_pair_potential_array = np.zeros((N_segments**2,2))
    grad_pair_potential_array_summed = np.array([0,0])
    ang_deriv_pair_potential = 0
    ang_deriv_pair_potential_array = np.zeros(N_segments**2)
    
    pos_part_a = np.zeros((N_segments,2))
    pos_part_b = np.zeros((N_segments,2))
    Diff = np.zeros((N_segments,len(pos_part_a),2))
    Diff_2 = np.zeros((N_segments,len(pos_part_a),2))
    
    pos_part_a[0] = pos_particle[a]
    pos_part_a[1:] = pos_particle[N_particle + a * (N_segments - 1):N_particle + a * (N_segments - 1) - 1 + N_segments]
    pos_part_b[0] = pos_particle[b]
    pos_part_b[1:] = pos_particle[N_particle + b * (N_segments - 1):N_particle + b * (N_segments - 1) - 1 + N_segments]
    
    orient_part_a = np.zeros(N_segments)
    angle_abs_com_segment_vec_array_a = np.zeros((N_segments,2))
    orient_part_a[0] = orient_particle[a]
    angle_abs_com_segment_vec_array_a[0] = angle_abs_com_segment_vec[a]
    orient_part_a[1:] = orient_particle[N_particle + a * (N_segments - 1):N_particle + a * (N_segments - 1) - 1 + N_segments]
    angle_abs_com_segment_vec_array_a[1:] = angle_abs_com_segment_vec[N_particle + b * (N_segments - 1):N_particle + b * (N_segments - 1) - 1 + N_segments]
    
    for i in range(N_segments):
        Diff_2[i] = np.minimum(abs(pos_part_a[i] - pos_part_b), (np.stack([[L,L]]*N_segments) - abs(pos_part_a[i] - pos_part_b)))
        Diff[i] = Diff_2[i] * np.sign(pos_part_a[i] - pos_part_b) * (- 2 * np.sign((abs(Diff_2[i])-abs(pos_part_a[i] - pos_part_b))%L) + 1)
    
    Abs_Diff = np.sqrt((np.concatenate(Diff)*np.concatenate(Diff)).sum(axis=1))
    
    grad_pair_potential_array =  np.concatenate(Diff)*(np.exp(- FallOff * Abs_Diff/Lambda)/(Abs_Diff**4/(Lambda**4)) * (2 + Abs_Diff/Lambda)/Lambda)[:,np.newaxis]
    
    grad_pair_potential_array_summed = - u0/(2 * N_segments**2) * (grad_pair_potential_array.sum(0))
    
    
    
    for i in range(N_segments):
        ang_deriv_pair_potential_array[i*N_segments:(i+1) * N_segments] = 2 * (np.exp(- FallOff * (Abs_Diff[i*N_segments:(i+1) * N_segments])/Lambda)/((Abs_Diff[i*N_segments:(i+1) * N_segments])**3/(Lambda**3)) * (2 + (Abs_Diff[i*N_segments:(i+1) * N_segments])/Lambda)/((Abs_Diff[i*N_segments:(i+1) * N_segments])/Lambda) * angle_abs_com_segment_vec_array_a[i,1] ) * ((np.concatenate(Diff)[i*N_segments:(i+1) * N_segments] * (np.stack((-np.sin(angle_abs_com_segment_vec_array_a[:,0]+orient_part_a),np.cos(angle_abs_com_segment_vec_array_a[:,0]+orient_part_a)),axis=-1)[i])).sum(1))
    
    ang_deriv_pair_potential = ang_deriv_pair_potential_array.sum(0)
                
    ang_deriv_pair_potential = - u0/(2 * N_segments**2)*ang_deriv_pair_potential
    
    
    return grad_pair_potential_array_summed,ang_deriv_pair_potential
def func_total_potentials(a,pos_particle,orient_particle):

    """
    For each a sum over all b closer than Cutoff and add up the pair potential contributions.
    """
    
    grad_total_potential_array = np.zeros((N_particle,2))
    ang_deriv_total_potential_array = np.zeros(N_particle)
    
    Cutoff = 2.4 * 2 * np.pi * np.sqrt(np.sin(OpeningAngle/4)**2/(OpeningAngle**2))
    
    for b in range(N_particle):
        if(a != b):
            if(np.sqrt((min(pos_particle[a][0] - pos_particle[b][0], L - (pos_particle[a][0] - pos_particle[b][0])))**2+(min(pos_particle[a][1]-pos_particle[b][1],L - (pos_particle[a][1]-pos_particle[b][1])))**2)<Cutoff):
                grad_total_potential_array[b],ang_deriv_total_potential_array[b]= func_pair_potentials(a,b,pos_particle,orient_particle)
    grad_total_potential = grad_total_potential_array.sum(0)
    ang_deriv_total_potential = ang_deriv_total_potential_array.sum(0)
    return grad_total_potential,ang_deriv_total_potential
def Dynamics():   
    """
    Updating of position. Compute the com of each particle, then update the com position and the angle of each crescent particle according to the eom and finally update every segment of each crescent particle according to what the new position and angle are
    """
    
    grad_total_potential = np.array([0,0])
    ang_deriv_total_potential = 0

            
            
    pos_com = np.zeros((N_Disks,2))   
    for i in range(N_particle):
        com_x = pos_particle[i][0] + Center_Of_Mass_Factor * np.cos(orient_particle[i])
        com_y = pos_particle[i][1] + Center_Of_Mass_Factor * np.sin(orient_particle[i])
        pos_com[i] = [com_x,com_y]
    for j in range(N_particle):
        pos_com[N_particle + j * (N_segments-1) : N_particle + (j + 1) * (N_segments-1)] = np.stack([pos_com[j]]*(N_segments-1))

    
    pos_particle_after = []
    orient_particle_after = []
    pos_com_after = []
    pos_particle_after = pos_particle.copy()
    orient_particle_after = orient_particle.copy()
    pos_com_after = pos_com.copy()
    
    
    for a in range(N_particle):
       
        grad_total_potential,ang_deriv_total_potential= func_total_potentials(a,pos_particle,orient_particle)
       
        Random_Number_Translation_x = np.random.normal(0,1)
        Random_Number_Translation_y = np.random.normal(0,1)
        Random_Number_Rotation = np.random.normal(0,1)
        
        for i in range(N_segments):
            pos_index = N_particle + a * (N_segments - 1) + i - 1
            
            if(i == 0):
                pos_com_after[a][0] += np.cos(orient_particle[a]) * v0 - grad_total_potential[0] + np.sqrt(2 * eta1) * Random_Number_Translation_x
                pos_com_after[a][1] += np.sin(orient_particle[a]) * v0 - grad_total_potential[1] + np.sqrt(2 * eta1) * Random_Number_Translation_y
                orient_particle_after[a] -= ang_deriv_total_potential + np.sqrt(2 * eta2) * Random_Number_Rotation
                
                if(Iteration_Counter%10 == 0):
                    writer.writerow(pos_particle[a])
                
                
            else:
                pos_com_after[pos_index][0] += np.cos(orient_particle[a]) * v0 - grad_total_potential[0] + np.sqrt(2 * eta1) * Random_Number_Translation_x
                pos_com_after[pos_index][1] += np.sin(orient_particle[a]) * v0 - grad_total_potential[1] + np.sqrt(2 * eta1) * Random_Number_Translation_y
                orient_particle_after[pos_index] -= ang_deriv_total_potential + np.sqrt(2 * eta2) * Random_Number_Rotation
            if(i == 0):
                pos_particle_after[a][0] = pos_com_after[a][0] + angle_abs_com_segment_vec[a][1] * np.cos(angle_abs_com_segment_vec[a][0] + orient_particle_after[a])
                pos_particle_after[a][1] = pos_com_after[a][1] + angle_abs_com_segment_vec[a][1] * np.sin(angle_abs_com_segment_vec[a][0] + orient_particle_after[a])
            else:
                pos_index = N_particle + a * (N_segments - 1) + i - 1
                pos_particle_after[pos_index][0] = pos_com_after[a][0] + angle_abs_com_segment_vec[pos_index][1] * np.cos(angle_abs_com_segment_vec[pos_index][0] + orient_particle_after[a])
                pos_particle_after[pos_index][1] = pos_com_after[a][1] + angle_abs_com_segment_vec[pos_index][1] * np.sin(angle_abs_com_segment_vec[pos_index][0] + orient_particle_after[a])
                
    pos_particle = pos_particle_after%L
    orient_particle = orient_particle_after
    
    
    Iteration_Counter += 1


def One_Run_Func(Run_Iteration,N_Disks):
    """
    Function for a single run.
    """
    np.random.seed()
    
    pos_particle = []
    orient_particle = []
    angle_abs_com_segment_vec = np.zeros((N_Disks,2))
    Center_Of_Mass_Factor = 0
    pos_particle,orient_particle,Center_Of_Mass_Factor = initilization()
    angle_abs_com_segment_vec = magnitude_angle_segments_array(Center_Of_Mass_Factor,pos_particle,orient_particle)
    
    f = open("Output_" + str(N_particle) + "/Position_" + str(N_particle) + "_" + str(Radius_Particles) + "_Run_" + str(Run_Iteration) + ".csv", 'w')
    writer = csv.writer(f)
    
    Iteration_Counter = 0
    
    for o in range(iterations):
        pos_particle,orient_particle = Dynamics(writer,Center_Of_Mass_Factor,angle_abs_com_segment_vec,pos_particle,orient_particle,Iteration_Counter)
        
        Iteration_Counter += 1
    
    

    

Run_Iteration = 0
if __name__ == '__main__':
    """
    Run the simulation. Use multiprocessing library to run on several cores.
    """
    
    os.mkdir("Output_" + str(N_particle))   

    n_cores = 4 #Modify to change number of cores used.

    pool = mp.Pool(processes=int(n_cores))

    res = pool.starmap(One_Run_Func,[(Run_Iteration,N_Disks) for Run_Iteration in range(Total_Number_Runs)])

    pool.close()

    pool.join() 