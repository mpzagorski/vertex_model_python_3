
###### Global constants of the system

global dt, viscosity, t_G1, t_G2, A_c, t_S, T1_eps, P, microns, time_hours, expansion_constant, L_tau, L_std, diff_rate_hours, pos_d, life_span, T1_newf
global IKNM_flag, DIVISION_flag

######MZ
global run_counter, time_initial, div_CPUtime, T1_CPUtimeA, T1_CPUtimeB, nT1, nT2, nD, time_unit
run_counter = 0		#measures number of iterations of run() function
time_initial = 0.	#assigned to time.time() when computation starts
div_CPUtime = 0. #for debugging
T1_CPUtimeA = 0.  #for debugging
T1_CPUtimeB = 0.  #for debugging
nT1 = 0.          #for debugging; #counts T1 transitions
nT2 = 0.          #for debugging; #counts T2 transitions
nD = 0          #for debugging;  #counts cell divisions
age_STD = 0.45    #corresponds to variance 0.2025; by default age_std = 0.2; MZ: 2020.16.04
T1_dict = {}      #for debugging; #counts number of occurences of T1 transition for a given edge
#_MAX_CELLS = 10**5   #defined in mesh.py
######

dt= 0.001            #time step
time_unit =1;        #unit_time = 1 means there are 1/dt time steps; unit_time = 10 means there are 10/dt steps between recording information events
viscosity= 0.02    	#viscosity*dv/dt = F #default #!!!can be changed when simulation is initiated!!!
T1_eps = 0.01       #default #!!!can be changed when simulation is initiated!!!
T1_newf = 1.01       #edge length after T1 transition = T!_newf * T1_eps!!! NOT IMPLEMENTED yet!!! Farfadihar has 1.05, Fletcher up to 1.5
#T1_eps = 0.         #switching of T1 transitions
A_c=1.3 #critical area
#A_c=1.3 #critical area
P= 0.0 #There is no boundary pressure in torous
diff_rate_hours=0.05 #differentiation rate (1/h) #!!!can be changed when simulation is initiated!!!

######noise related
L_tau = 100/ (13.*60)  #0.128205, roughly 1 minute when 13h corresponds to 100 unit steps #lag in auto-correlation function of O-U noise in Lambda values
L_std = 0.01        #standard deviation in Lambda values

######switching simulation mode (no all combinations tested)
IKNM_flag = 1       #1: Interkinetic nuclear movement is ON, 0: noIKNM, A_0 is set to 1
DIVISION_flag = 1   #1: divisions are switche ON, 0: no divisions

"""60 hours Dorsal"""
pos_d="Dorsal"
J=60
t_G1=0.4            		#Time proportion in G1 phase
t_S = 0.5*(2.0/3.0)            #Time proportion in S phase
t_G2 = 0.5*(1.0/3.0)           #Time proportion in G2 phase 
t_M=1.0-(t_G1+t_S+t_G2)

experimental_perimeter = 9.93 #average, Dorsal e10.5
simulation_perimeter = 2.17  #average, Dorsal 60 hours
microns = experimental_perimeter/simulation_perimeter 
experimental_cell_cycle = 13.0 #average, hours 60 hours
simulation_cell_cycle = 100.0 #average step units 
life_span = simulation_cell_cycle
time_hours = experimental_cell_cycle/simulation_cell_cycle 

expansion_constant = 50     #sets proportion in expansion between AP and DV axis #!!!can be changed when simulation is initiated!!!
