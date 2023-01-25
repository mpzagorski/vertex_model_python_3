
# coding: utf-8

# In[6]:

import sys
import os
import numpy as np
import time
from run_select_noise import run_simulation_INM_screen, check_estage_detailed, set_viscosity, set_T1_eps, set_expansion_constant, set_diff_rate_hours, set_life_span, set_time_unit, get_viscosity, get_T1_eps, get_expansion_constant, get_diff_rate_hours, get_time_unit, get_life_span
from run_select_noise import set_OU_noise, set_flags, set_age_STD

#### Parameter used in the simulation are picked up form Gobal_Constant.py file or specified below

####Order of parameters in the command line: #after 2020.07.22
#K G L sim_type viscosity T1_eps expansion_constant diff_rate_hours life_span time_start time_end time_unit Lambda_tau Lambda_std age_STD random_seed
#sys.argv = [sys.argv[0], 1.0, 0.12, -0.074, 0, 0.02, 0.01, 50, 0, 100, 11, 21, 10, 0.1282, 0.01, 0.45, 12348, "OUT"] #comment out when running from a command line :)
#typical command line: python3 main_run_simulation.py 1.0 0.12 -0.074 0 0.02 0.01 50 0 100 11 21 10 0.1282 0.01 0.45 12348 ./OUT

print(len(sys.argv))    
IKNM_flag_new = 1       #IKNM switched ON (1) or OFF (0)
DIVISION_flag_new = 1   #Divisions switched ON (1) or OFF (0)

if len(sys.argv) == 1:  #if no cmd line entry is provided
    
    K, G, L = 1.0, 0.12, -0.074    #region C
    #K, G, L = 1.0, 0.12, -0.393    #region B
    #K, G, L = 1.0, 0.12, -0.711     #region A
    
    type_ = 0
    viscosity_new, T1_eps_new, expansion_constant_new, diff_rate_hours_new = 0.02, 0.01, 50, 0.
    
    life_span_new = 100
    #time_start, time_end, time_unit_new = 201, 301, 100
    #time_start, time_end, time_unit_new = 201, 301, 1 #use for movies    
    time_start, time_end, time_unit_new = 11, 21, 10  #use for debug
    
    L_tau_new = 0.1282
    L_std_new = 0.02
    
    age_STD_new = 0.45
    random_seed = 12345   #any number < 2^32 will do
    output_dir_name = "OUT"     #default output directory
    
elif len(sys.argv) == 18:
    K, G, L,type_, viscosity_new, T1_eps_new, expansion_constant_new, diff_rate_hours_new, life_span_new, time_start, time_end, time_unit_new, L_tau_new, L_std_new, age_STD_new, random_seed, output_dir_name = sys.argv[1:]
    # 1.0, 0.04, 0.075, 0, 0.02, 0.01, 50, 0.05, 10, 10, 1299709, OUT_HISTORY
    K = float(K); G = float(G); L = float(L);
    type_ = int(type_)
    viscosity_new = float(viscosity_new); T1_eps_new =float(T1_eps_new); expansion_constant_new = float(expansion_constant_new); diff_rate_hours_new = float(diff_rate_hours_new); life_span_new = float(life_span_new)
    time_start = float(time_start); time_end = float(time_end); time_unit_new = float(time_unit_new); L_tau_new = float(L_tau_new); L_std_new = float(L_std_new); age_STD_new = float(age_STD_new); random_seed = int(random_seed)
else:
    print("len(sys.argv) = %d does not match number of command line parameters = 18. Simulation aborted." % len(sys.argv))
    sys.exit("Mismatch in number of command line parameters")

#### Setting new parameter values
set_viscosity(viscosity_new)
set_T1_eps(T1_eps_new)
set_OU_noise(L_tau_new,L_std_new)
set_flags(IKNM_flag_new, DIVISION_flag_new)
set_age_STD(age_STD_new)

set_expansion_constant(expansion_constant_new)
set_diff_rate_hours(diff_rate_hours_new)
set_life_span(life_span_new)
set_time_unit(time_unit_new)

#run simulation with the choosen parameters
rand =  np.random.RandomState(random_seed) #random number to choose Lambda
params = [K,G,L]  # K=x[0],G=x[1],L=x[2]

#params0 = params                #parameters for the 1st stage are the same as for the 2nd stage
params0 = [1, 0.07, -0.184]      #parameters for the 1st stage

nX0 = 10        #initial cells left-rigt
nY0 = 10        #initial cells bottom-up

time0 = time.time()         #sets timer for the computation

#### Saving history structure
if not os.path.exists(output_dir_name): # if the folder doesn't exist create it
    os.makedirs(output_dir_name)
output_dir_name = output_dir_name+'/'

#### Exporting history to files
time_unit = get_time_unit()
suffix = "type" + str(type_) + "_G" + str(G) + "_L" + str(L) + "_v" + str(get_viscosity()) + "_ec" + str(get_expansion_constant()) + "_pr" + str(get_life_span()) + "_id" + str(random_seed)
time_tot = (time_start + time_end) / time_unit
file_suffix = str(int(round(time_start-1))) + "-" + str(int(round(time_tot * time_unit))) + "-" + str(int(round(time_unit))) + "_" + suffix + ".dat"

#### Opening log file for
log_file_name = output_dir_name + "/log_0-" + file_suffix
print(log_file_name)
fout = open(log_file_name, "w")   #opens file
fout.write("timeStep\t cellsAll\t cellsNonEmpty\t DVlen\t APlen\t APDVratio\t nT1\t nT2\t nD\t CPUtime \n")

#### Opening log file for cell division ids 
div_file_name = output_dir_name + "/divisionEXT_0-" + file_suffix
print(div_file_name)
foutD = open(div_file_name, "w")   #opens file
foutD.write("timeStep\t ndiv_pMN\t ndiv_pD\t time_division\t vertices_edge \n")

#### Opening log file for cell division ids and daughter dis
lineage_file_name = output_dir_name + "/lineageEXT_0-" + file_suffix
print(lineage_file_name)
foutDL = open(lineage_file_name, "w")   #opens file
foutDL.write("timeStep\t ndiv_pMN\t ndiv_pD\t cell_ids\t daughter_ids \n")

#### Opening log file for T1 dictionary  #allows to track oscillatory T1 transitions
T1ids_file_name = output_dir_name + "/tOneDict_0-" + file_suffix
print(T1ids_file_name)
foutDT = open(T1ids_file_name, "w")   #opens file
foutDT.write("timeStep\t T1_unique\t edge_1\t edge_2\t nT1\n")

#### Opening log file for T1 ids
T1_file_name = output_dir_name + "/tOneEXT_0-" + file_suffix
print(T1_file_name)
foutT1 = open(T1_file_name, "w")   #opens file
foutT1.write("timeStep\t nT1s\t nT2s_within_nT1\n")    #basic T1 and T2 file

#### Running SIMULATION
print("This is the name of the script: ", sys.argv[0])
print("Random_seed = %d\n" % random_seed)
check_estage_detailed(params, type_)    #printing parameters


history_start, history_end = run_simulation_INM_screen(params, params0, nX0, nY0, time_start, time_end, rand, type_, fout, foutD, foutDL, foutDT, foutT1) #return hist
history = history_start+history_end

time1 = time.time()-time0   #returns computation time
print("Computation time: %g [s]\n" % round(time1,2))

fout.close()    #closing log file
foutD.close()   #closing division log file
foutDL.close()   #closing lineage log file
foutDT.close()   #closing T1 ids log file
foutT1.close()  #closing T1 log file

#### Saving mesh and other properties

#Parameters
output_file_dat = output_dir_name + "/parameters_0-" + file_suffix
fout = open(output_file_dat, "w")   #opens file
fout.write("K\t G\t L\t type_\t viscosity\t T1_eps\t expansion_constant\t diff_rate_hours\t life_span\t time_start\t time_end\t time_unit\t L_tau\t L_std\t age_STD\t random_seed\n")
fout.write("%g\t %g\t %g\t %d\t %g\t %g\t %g\t %g\t %g\t %d\t %d\t %d\t %g\t %g\t %g\t %d\n" % (K, G, L,type_, viscosity_new, T1_eps_new, expansion_constant_new, diff_rate_hours_new, life_span_new, time_start, time_end, time_unit, L_tau_new, L_std_new, age_STD_new, random_seed))
fout.close()

history_len = len(history)
time_tot = history_len

#Vertices
output_file_dat = output_dir_name + "/vertices_0-" + file_suffix
fout = open(output_file_dat, "w")   #opens file
for id in range(time_tot):
    tempV = np.transpose(history[id].mesh.vertices)
    ltab = len(tempV)
    fout.write("generation\t%d\t%d\n" % (id * time_unit,ltab))
    for v in tempV:
        fout.write("%g\t%g\n" % (v[0],v[1])),
fout.close()

#Edge vectors
output_file_dat = output_dir_name + "/edges_0-" + file_suffix
fout = open(output_file_dat, "w")   #opens file
for id in range(time_tot):
    tempE = np.transpose(history[id].mesh.edge_vect)
    ltab = len(tempE)
    fout.write("generation\t%d\t%d\n" % (id * time_unit,ltab))
    for e in tempE:
        fout.write("%g\t%g\n" % (e[0],e[1])),
fout.close()

#Edge faces
output_file_dat = output_dir_name + "/faces_0-" + file_suffix
fout = open(output_file_dat, "w")   #opens file
for id in range(time_tot):
    tempF = np.transpose(history[id].mesh.face_id_by_edge)
    ltab = len(tempF)
    fout.write("generation\t%d\t%d\n" % (id * time_unit,ltab))
    for n in tempF:
        fout.write("%d\n" % n),
fout.close()

#Edge rotation (next table)
output_file_dat = output_dir_name + "/edges_next_0-" + file_suffix
fout = open(output_file_dat, "w")   #opens file
for id in range(time_tot):
    tempE = np.transpose(history[id].mesh.edges.next)
    ltab = len(tempE)
    fout.write("generation\t%d\t%d\n" % (id * time_unit,ltab))
    for e in tempE:
        fout.write("%d\n" % e),
fout.close()

#Edge reversed id
output_file_dat = output_dir_name + "/edges_reverseID_0-" + file_suffix
fout = open(output_file_dat, "w")   #opens file
for id in range(time_tot):
    tempE = np.transpose(history[id].mesh.edges.reverse)
    ltab = len(tempE)
    fout.write("generation\t%d\t%d\n" % (id * time_unit,ltab))
    for e in tempE:
        fout.write("%d\n" % e),
fout.close()

#Area
output_file_dat = output_dir_name + "/area_0-" + file_suffix
fout = open(output_file_dat, "w")   #opens file
for id in range(time_tot):
    tempF = np.transpose(history[id].mesh.area)
    ltab = len(tempF)
    fout.write("generation\t%d\t%d\n" % (id * time_unit,ltab))
    for a in tempF:
        fout.write("%g\n" % a),
fout.close()

#Exporting history to files -- continued

#Parent and daughters (the same numbers)
output_file_dat = output_dir_name + "/parents_0-" + file_suffix
fout = open(output_file_dat, "w")   #opens file
for id in range(time_tot):
    tempP = np.transpose(history[id].properties['parent'])
    ltab = len(tempP)
    fout.write("generation\t%d\t%d\n" % (id * time_unit,ltab))
    for p in tempP:
        fout.write("%d\n" % p),
fout.close()

#Parent group (0 for pd0, 1 for pMN, 2 for pd2)
output_file_dat = output_dir_name + "/parents_group_0-" + file_suffix
fout = open(output_file_dat, "w")   #opens file
for id in range(time_tot):
    tempP = np.transpose(history[id].properties['parent_group'])
    ltab = len(tempP)
    fout.write("generation\t%d\t%d\n" % (id * time_unit,ltab))
    for p in tempP:
        fout.write("%d\n" % p),
fout.close()

#Age
output_file_dat = output_dir_name + "/age_0-" + file_suffix
fout = open(output_file_dat, "w")   #opens file
for id in range(time_tot):
    tempA = np.transpose(history[id].properties['age'])
    ltab = len(tempA)
    fout.write("generation\t%d\t%d\n" % (id * time_unit,ltab))
    for p in tempA:
        fout.write("%g\n" % p),
fout.close()

#Force in X and Y direction (per edge) !!!NO force file exported to save space
#output_file_dat = output_dir_name + "/forceXY_0-" + file_suffix
#fout = open(output_file_dat, "w")   #opens file
#for id in range(time_tot):
#    tempX = np.transpose(history[id].properties['force_x'])
#    tempY = np.transpose(history[id].properties['force_y'])
#    ltab = len(tempX)
#    fout.write("generation\t%d\t%d\n" % (id * time_unit,ltab))
#    for k in range(ltab):
#        fout.write("%g\t%g\n" % (tempX[k], tempY[k])),
#fout.close()

print("Done\n")
