
# # All the function to run a simulation

#Import libraries

#########################################################################PARALELL################################################

#########################################################################PARALELL################################################
import itertools
import numpy as np
import mesh as model
import initialisation as init
from forces import TargetArea, Tension, Perimeter, Pressure
import os
import warnings
import time
warnings.filterwarnings('ignore') #Don't show warnings
from Gobal_Constant import dt, viscosity, t_G1, t_G2, t_S, A_c, J, pos_d, T1_eps, P, microns, time_hours, expansion_constant, diff_rate_hours, run_counter, time_initial, div_CPUtime, T1_CPUtimeA, T1_CPUtimeB, nT1, nT2, nD, time_unit, life_span, T1_newf, T1_dict #file with necessary constants
from Gobal_Constant import IKNM_flag, DIVISION_flag, L_tau, L_std, age_STD


def set_viscosity(viscosity_new=0.02):
    global viscosity
    viscosity = viscosity_new

def set_T1_eps(T1_eps_new=0.01):
    global T1_eps
    T1_eps = T1_eps_new

def set_expansion_constant(expansion_constant_new=50):
    global expansion_constant
    expansion_constant = expansion_constant_new

def set_diff_rate_hours(diff_rate_hours_new=0.05):  #differentiation rate (1/h) 
    global diff_rate_hours
    diff_rate_hours = diff_rate_hours_new
    
def set_flags(IKNM_flag_new=1, DIVISION_flag_new=1):
    global IKNM_flag, DIVISION_flag
    IKNM_flag = IKNM_flag_new
    DIVISION_flag = DIVISION_flag_new

def set_OU_noise(L_tau_new=1., L_std_new=0.01):
    global L_tau, L_std
    L_tau = L_tau_new
    L_std = L_std_new
    
def set_age_STD(age_STD_new=0.45):
    global age_STD
    age_STD = age_STD_new

def set_time_unit(time_unit_new=1):
    global time_unit
    time_unit = time_unit_new

def set_life_span(life_span_new=1):
    global life_span
    life_span = life_span_new

def set_T1_newf(T1_newf_new=1):
    global T1_newf
    T1_newf = T1_newf_new

def get_viscosity():
    global viscosity
    return viscosity

def get_T1_eps():
    global T1_eps
    return T1_eps

def get_expansion_constant():
    global expansion_constant
    return expansion_constant

def get_diff_rate_hours():  #differentiation rate (1/h) 
    global diff_rate_hours
    return diff_rate_hours

def get_time_unit():
    global time_unit
    return time_unit

def get_life_span():
    global life_span
    return life_span

def get_T1_newf():
    global T1_newf
    return T1_newf
    
def check_estage():
    print("Running a %s hours" % J) #+ " %s"%pos_d_v
    print("dt = %s" % dt)            #time step
    print("viscosity = %s" % viscosity)  #viscosity*dv/dt = F
    print("A_c = %s" % A_c) #critical area
    print("T1_eps = %s" % T1_eps)
    
def check_estage_detailed(x,type_sim):
    K=x[0]
    G=x[1]
    L=x[2]
    print("Running a %s hours" % J) #+ " %s"%pos_d_v
    print("dt = %s" % dt)            #time step
    print("viscosity = %s" % viscosity)  #viscosity*dv/dt = F
    print("A_c = %s" % A_c) #critical area
    print("T1_eps = %s" % T1_eps)
    print("T1_newf = %s" % T1_newf)
    print("Life_span = %s" % life_span)
    print("Simulation type = %d" % type_sim)
    if type_sim == 0:
        print("Differentiation rate = 0 (type 0)")
    else:
        print("Differentiation rate = %s" % diff_rate_hours)
    print("Expansion constant = %s" % expansion_constant)
    print("K = %s,\tGamma = %s,\tLambda = %s\n" % (K,G,L))

# run simulation
def run(simulation,N_step,skip):
    return [cells.copy() for cells in itertools.islice(simulation,0,N_step,skip)]


def division_axis(mesh,face_id,rand):
    """Choose a random division axis (given as a pair of boundary edges to bisect) for the given cell.
    
    The first edge is chosen randomly from the bounding edges of the cell with probability proportional 
    to edge length. The second edge is then fixed to be n_edge/2 from the first. 
    """
    edges = mesh.boundary(face_id)
    if edges==[-1]:
        print('here')
        os._exit(1)
    p = np.cumsum(mesh.length[edges])
    e0 = p.searchsorted(rand.rand()*p[-1])
    return edges[e0],edges[e0-len(edges)//2]  

def bin_by_xpos(cells,percentiles):
    vx = cells.mesh.vertices[0]
    #simple 'midpoint' as mean of vertex positions
    mid_x = np.bincount(cells.mesh.face_id_by_edge,weights=vx)
    counts = np.maximum(np.bincount(cells.mesh.face_id_by_edge),1.0)
    mid_x = mid_x / counts 
    width = cells.mesh.geometry.width
    return np.searchsorted(percentiles,(mid_x/width + 0.5) % 1.0)   

#simulation without division #old implementation
def basic_simulation(cells,force,dt=dt,T1_eps=0.04):
    while True:
        cells.mesh , number_T1 = cells.mesh.transition(T1_eps)
        F = force(cells)/viscosity
        expansion = 0.05*np.average(F*cells.mesh.vertices,1)*dt
        dv = dt*model.sum_vertices(cells.mesh.edges,F) 
        cells.mesh = cells.mesh.moved(dv).scaled(1.0+ expansion)
        yield cells

def write_to_division_extended_file(foutD, run_counter_by_step, n_pMN, n_pD, time_div, edge_division):
    ####print division to file###; from 2019.04.12 division ids are in lineage file
    foutD.write("%d %d %d " % (run_counter_by_step, n_pMN, n_pD))
    for time_stamp in time_div:
        foutD.write("%g " % time_stamp)
    for edge_ver in edge_division:
        foutD.write("%g " % edge_ver)
    foutD.write("\n")
    
def write_to_division_lineage_file(foutDL, run_counter_by_step, n_pMN, n_pD, ids_div, ids_daughters):
    ####print division to file###
    foutDL.write("%d %d %d " % (run_counter_by_step, n_pMN, n_pD))
    for id_cell in ids_div:                 #from 2019.04.12 division ids are in lineage file
        foutDL.write("%d " % int(id_cell))
    for id_cell in ids_daughters:
        foutDL.write("%d " % int(id_cell))
    foutDL.write("\n")
    
def write_to_T1_cell_ids_file(foutDT, run_counter_by_step, n_T1, n_T2, ids_T1, ids_T2):
    ####print T1 and T2 ids to file###
    foutDT.write("%d %d %d " % (run_counter_by_step, n_T1, n_T2))
    for id_cell in ids_T1:                 #(A, B, C, D), 4 cell ids before T1 transition; A neighbours B; after T1: C neighbours D
        foutDT.write("%d " % int(id_cell))
    for id_cell in ids_T2:                 #id of two-sided cell that is removed
        foutDT.write("%d " % int(id_cell))
    foutDT.write("\n")
    
def write_to_T1_dict_file(foutDT, run_counter_by_step, T1_dict):
    ####print T1 dictionary to file###
    foutDT.write("%d %d " % (run_counter_by_step, len(T1_dict)))
    if len(T1_dict):
        for items in T1_dict.items():
            foutDT.write("%d %d %d " % (items[0][0], items[0][1], items[1]))
    foutDT.write("\n")            
            
def write_to_T1_extended_file(foutT1, run_counter_by_step, n_T1, n_T2, time_T1, time_remove, mid_point_T1, edge_double_remove):
    ####print T1 and T2 to file###
    #foutT1.write("%d %d %d " % (run_counter/time_step, int(nT1), int(nT2))) #for basic file
    foutT1.write("%d %d %d " % (run_counter_by_step, int(n_T1), int(n_T2)))
    for time_stamp in time_T1:
        foutT1.write("%g " % time_stamp)
    for time_stamp in time_remove:
        foutT1.write("%g " % time_stamp)
    for mid_point in mid_point_T1:
        foutT1.write("%g " % mid_point)
    for edge_ver in edge_double_remove:
        foutT1.write("%g " % edge_ver)
    foutT1.write("\n")

def get_noise_edges(cells, L_std, rand):        #MZ fills cells.properties['Lambda_edge'] with random numbers N(L, L_std)
    properties = cells.properties
    edges = cells.mesh.edges
    half_edges_1 = edges.ids[edges.ids < edges.reverse]  # half the edges have id < reverse_id
    half_edges_2 = edges.reverse[half_edges_1]
    nh_edges = len(half_edges_1)
    #n_edges = len(edges.ids)
    #print("Length n_edges=%d, nh_edges=%d" % (n_edges,nh_edges))
    L_av = properties['Lambda']
    tempL = rand.normal(L_av, L_std, nh_edges)
    properties["Lambda_edge"][half_edges_1] = tempL
    properties["Lambda_edge"][half_edges_2] = tempL

def get_OU_noise_edges(cells, L_std, tau, dt, rand):        #MZ fills cells.properties['Lambda_edge'] with random numbers N(L, L_std)
    properties = cells.properties
    edges = cells.mesh.edges
    if L_std > 0:
        half_edges_1 = edges.ids[edges.ids < edges.reverse]  # half the edges have id < reverse_id
        half_edges_2 = edges.reverse[half_edges_1]
        nh_edges = len(half_edges_1)
        #n_edges = len(edges.ids)
        #print("Length n_edges=%d, nh_edges=%d" % (n_edges,nh_edges))
        L_half1 = properties['Lambda_edge'][half_edges_1]
        L_av = properties['Lambda']
        L_av_tab = np.full(nh_edges, L_av)
        L_noise = rand.normal(0, 1, nh_edges)
        #noise updated with OU process
        tempL = L_half1 - (dt/tau)*(L_half1 - L_av_tab) + np.sqrt(2 * L_std**2 *dt/tau) * L_noise #MZ: Ornstein-Uhlenbeck discretizaiton that follows Curran et al., Dev Cell, 2017# tested independently in Mathematica
        properties["Lambda_edge"][half_edges_1] = tempL     #new noise assigned to "Lambda_edge"
        properties["Lambda_edge"][half_edges_2] = tempL     #new noise assigned to "Lambda_edge"
    else:                                                   #no noise
        n_edges = len(edges)
        L_av = properties['Lambda']
        tempL = np.full(n_edges, L_av)
        properties["Lambda_edge"] = tempL 

def append_noise_edges(cells, L_std, nnew_edges, rand):        #MZ appends cells.properties['Lambda_edge'] with random numbers N(L, L_std)
    properties = cells.properties    
    edges = cells.mesh.edges
    if L_std > 0:
        L_av = properties['Lambda']
        tempL = rand.normal(L_av, L_std, nnew_edges)
        properties["Lambda_edge"] = np.concatenate([properties["Lambda_edge"], tempL]) #MZ appending new values
        half_edges_1 = edges.ids[edges.ids < edges.reverse]  # half the edges have id < reverse_id
        half_edges_2 = edges.reverse[half_edges_1]
        properties["Lambda_edge"][half_edges_2] = properties["Lambda_edge"][half_edges_1] #MZ make sure that after division(s) all edges and the reverse edges have the same lambda
    else:
        L_av = properties['Lambda']
        tempL = np.full(nnew_edges, L_av)
        properties["Lambda_edge"] = np.concatenate([properties["Lambda_edge"], tempL])

# simulation with division, INM and with/without differentiation rate
def simulation_with_division_TYPE_noise(SIM_TYPE,cells,force,fout,foutD,foutDL,foutDT,foutT1,dt=dt,T1_eps=T1_eps,lifespan=life_span,rand=None):
    global run_counter, time_initial, div_CPUtime, T1_CPUtimeA, T1_CPUtimeB, nT1, nT2, nD, age_STD, L_std, L_tau, IKNM_flag, DIVISION_flag, T1_dict
    time_step = int(1.0/dt)
        
    properties = cells.properties
    properties['parent'] = cells.mesh.face_ids #save the ids to control division parents-daugthers 
    properties['ageingrate'] = rand.normal(1.0/lifespan,age_STD/lifespan,len(cells)) #degradation rate per each cell #MZ: np.random -> rand
                                                                                     #age_STD=0.45 corresponds to variance 0.2025; before age_std = 0.2; MZ: 2020.16.04
    properties['ids_division'] = [] #save ids of the cell os the division when it's ready per each time step
    if SIM_TYPE == 1:
        properties['poisoned'] = np.zeros(len(cells)) ### to add diferenciation rate in PMN
    properties['force_x'] = []
    properties['force_y'] = []
    #properties['T1_angle'] = []
    #properties['Division_angle'] = []
    properties['time_stamp_division'] = []  #MZ division tracking
    properties['time_stamp_T1'] = []  #MZ T1 tracking
    properties['time_stamp_remove'] = []  #MZ remove (T2, double edge) tracking
    properties['edge_division'] = []  #MZ division tracking
    properties['mid_point_T1'] = []  #MZ T1 tracking
    properties['edge_double_remove'] = []  #MZ remove tracking
    properties['ids_daughters'] = [] #MZ daughter cell ids
    properties['T1_ids'] = [] #MZ T1 cell ids
    properties['T2_ids'] = [] #MZ ids of removed cells
    
    expansion = np.array([0.0,0.0])
    while True:
        if DIVISION_flag==1:
            #cells id where is true the division conditions: living cells & area greater than 2 & age cell in mitosis 
            ready = np.where(~cells.empty() & (cells.mesh.area>=A_c) & (cells.properties['age']>=(t_G1+t_S+t_G2)))[0]  
            if len(ready): #these are the cells ready to undergo division at the current timestep
                properties['ageingrate'] =np.append(properties['ageingrate'], np.abs(rand.normal(1.0/lifespan,age_STD/lifespan,2*len(ready)))) #Zagorski; 2.0 -> 2 #MZ: np.random -> rand
                properties['age'] = np.append(properties['age'],np.zeros(2*len(ready)))
                properties['parent'] = np.append(properties['parent'],np.repeat(properties['parent'][ready],2))  # Daugthers and parent have the same ids
                properties['parent_group'] = np.append(properties['parent_group'],np.repeat(properties['parent_group'][ready],2)) #can be used to draw clones
                if SIM_TYPE == 1:
                    properties['poisoned'] = np.append(properties['poisoned'], np.zeros(2*len(ready))) ### to add diferenciation rate in PMN
                properties['ids_division'] = np.append(properties['ids_division'], ready)
                properties['time_stamp_division'] = np.append(properties['time_stamp_division'],np.full(len(ready),run_counter*dt))        #MZ debugging
                
                #division
                #print("Division occurs")
                edge_pairs = [division_axis(cells.mesh,cell_id,rand) for cell_id in ready] #New edges after division 
                cells.mesh, ids_daughters = cells.mesh.add_edges_extended(edge_pairs) #Add new edges in the mesh
                properties['ids_daughters'] = np.append(properties['ids_daughters'], ids_daughters)
                
                #lambda_noise (after div)
                nnew_edges = 6*len(ready)
                append_noise_edges(cells, L_std, nnew_edges, rand)

                #for cell_id in ready:
                #    print(cell_id,properties['age'][cell_id])         
                #    fout.write("%d\t%g\n" % (cell_id,properties['age'][cell_id]))      #getting cell age at division
                for i in range(len(ready)):
                    commun_edges = np.intersect1d(cells.mesh.length[np.where(cells.mesh.face_id_by_edge==(cells.mesh.n_face-2*(i+1)))[0]],cells.mesh.length[np.where(cells.mesh.face_id_by_edge==(cells.mesh.n_face-1-2*i))[0]])
                    division_new_edge=np.where(cells.mesh.length==np.max(commun_edges))[0]
                    #properties['Division_angle_pD']= np.append(properties['Division_angle_pD'],cells.mesh.edge_angle[division_new_edge][0])
                    properties['edge_division'] = np.append(properties['edge_division'],cells.mesh.vertices.T[ division_new_edge ]) #vertices of edge(s) after division(s)
        else: #Disions are switched off with DIVISION_flag
            ready = []
        
        if SIM_TYPE == 1:
            ###### Defferentiation rate
            properties['differentiation_rate'] = time_hours*dt*(np.array([diff_rate_hours,diff_rate_hours,diff_rate_hours]))[properties['parent_group']] #Used 0.02, 0.0002 & 1/13
            properties['poisoned'] = properties['poisoned'] - (properties['poisoned']-1) * (~(cells.empty()) & (rand.rand(len(cells)) < properties['differentiation_rate']))
        
        """Calculate z nuclei position (Apical-Basal movement), depending of the cell cycle phase time and age of the cell"""
        properties['age'] = properties['age']+dt*properties['ageingrate'] #add time step depending of the degradation rate 
        if IKNM_flag==1: #IKNM is turned on
            N_G1=1-1.0/t_G1*properties['age'] #nuclei position in G1 phase
            N_S=0
            N_G2=1.0/(t_G2)*(properties['age']-(t_G1+t_S))  #nuclei position in G2 and S phase
            properties['zposn'] = np.minimum(1.0,np.maximum(N_G1,np.maximum(N_S,N_G2)))
       
        ############Clock############
        dtime_T1 = time.time()      
        #######Clock Continued####### 
        
        """Target area function depending age and z nuclei position"""
        if IKNM_flag==1: ###with IKNM
            if SIM_TYPE == 1:       ###with differentiation
                properties['A0'] = (properties['age']+1.0)*0.5*(1.0+properties['zposn']**2)*(1.0-cells.properties['poisoned'])
            else:                   ###without differentiation
                properties['A0'] = (properties['age']+1.0)*0.5*(1.0+properties['zposn']**2)
        else:            ###without IKNM
            if SIM_TYPE == 1:   ###with differentiation
                properties['A0'] = np.full(len(cells),1.)*(1.0-cells.properties['poisoned'])
            else:               ###without differentiation
                #Target area constant or with fluctuations
                properties['A0'] = np.full(len(cells),1.)
                # noise_dt = 10
                # A0_std = 0.3
                # if (run_counter % noise_dt) == 0:
                #     properties['A0'] = rand.normal(1.0, A0_std, len(cells))       #noise updated every noise_dt x dt steps with A0_std deviation
        
        ########T1 transitions##########
        #cells.mesh , number_T1 = cells.mesh.transition(T1_eps)
        #cells.mesh, number_T1, T1_counter, T2_counter, mid_point_T1, two_sided_ver, T1_cell_ids, T2_cell_ids = cells.mesh.transition_count_extended(T1_eps)  #check edges verifing T1 transition #used before 2019.04.16
        #cells.mesh, number_T1, T1_counter, T2_counter, mid_point_T1, two_sided_ver, T1_cell_ids, T2_cell_ids, T1_edge = cells.mesh.transition_count_extended_track(T1_eps)  #check edges verifing T1 transition #introduced 2020.04.29
        #cells.mesh, number_T1, T1_counter, T2_counter, T1_edge = cells.mesh.transition_count_T1_dict(T1_eps)  #check edges verifing T1 transition #introduced 2020.04.16
        cells.mesh, number_T1, T1_counter, T2_counter, T1_edge, to_del = cells.mesh.transition_count_T1_dict_del(T1_eps)  #check edges verifing T1 transition #introduced 2020.04.16
        #cells.mesh, number_T1, T1_counter, T2_counter = cells.mesh.transition_count(T1_eps)  #check edges verifing T1 transition #used before 2019.04.16
        
        if len(T1_edge):
            for e0, e3 in zip(T1_edge[0::2],T1_edge[1::2]):  #e0 & e3 udergo T1 transition
                #print("e0, e3: ", e0, e3)
                T1_dict[(e0, e3)] = T1_dict.get((e0,e3), 0) + 1
                #print("T1_ids: ", T1_cell_ids)
                
        # if len(mid_point_T1):
        #     properties['time_stamp_T1'] = np.append(properties['time_stamp_T1'],np.full(len(mid_point_T1),run_counter*dt))        #MZ debugging
        #     properties['mid_point_T1'] = np.append(properties['mid_point_T1'],mid_point_T1)
        #     properties['T1_ids'] = np.append(properties['T1_ids'],T1_cell_ids)
            
        # if T2_counter:
        #     properties['time_stamp_remove'] = np.append(properties['time_stamp_remove'],np.full(int(T2_counter),run_counter*dt))        #MZ debugging
        #     properties['edge_double_remove'] = np.append(properties['edge_double_remove'],two_sided_ver)
        #     properties['T2_ids'] = np.append(properties['T2_ids'],T2_cell_ids)
        #     #print("T2_ids: ", T2_cell_ids)

        # if len(mid_point_T1) != T1_counter:
        #     print('Warning!!!!! Unequal len(mid_point_T1) != T1_counter') #Should never take place
        # if len(two_sided_ver) != 4*T2_counter:
        #     print('Warning!!!!! Unequal len(mid_point_T2) != T2_counter') #Should never take place
        
        ########Noise update#########
        #if len(to_keep) < len(cells.mesh.face_id_by_edge):
        if len(to_del) > 0:
            properties['Lambda_edge'] = np.delete(properties['Lambda_edge'],to_del)            
        
        #get_noise_edges(cells, L_std, rand)       #MZ fills cells.properties['Lambda_edge'] with random numbers N(L, L_std)
        get_OU_noise_edges(cells, L_std, L_tau, dt, rand)
        
        ############Clock############
        dtime_T1 = time.time() - dtime_T1    
        dtime_T1B = time.time() 
        T1_CPUtimeA += dtime_T1;
        nT1 += T1_counter
        nT2 += T2_counter
        nD  += len(ready)
        #######Clock End#######
        #edges = cells.mesh.edges
        #print("ids: ", edges.ids)
        #print("reverse: ", edges.reverse)
        
        F = force(cells)/viscosity  #force per each cell force= targetarea+Tension+perimeter+pressure_boundary 
        dv = dt*model.sum_vertices(cells.mesh.edges,F) #movement of the vertices using eq: viscosity*dv/dt = F
        properties['force_x'] = F[0]*viscosity
        properties['force_y'] = F[1]*viscosity
        
        ############Clock############
        dtime = time.time() - time_initial      #MZ
        dtime_T1B = time.time() - dtime_T1B    #MZ
        T1_CPUtimeB += dtime_T1B;
        if (run_counter % time_step) == 0:          
            #print("%d,\tcells: %d,\t#T1: %g,\tT2: %g,\tT1_check: %g,\tT1_done: %g,\tTime: %g" % (run_counter/time_step, cells.mesh.n_face, float(nT1)/time_step, float(nT2)/time_step, T1_CPUtimeA, T1_CPUtimeB, dtime))  #MZ
            print("%d,\tcells: %d,\tnon-empty: %d, \t#T1: %g,\tT2: %g, \tTime: %g" % (run_counter/time_step, cells.mesh.n_face, sum(~cells.empty()), float(nT1)/time_step, float(nT2)/time_step, dtime))  #MZ
            #print("T1_eps: %g, exapns_const: %g\n" % (T1_eps, expansion_constant))    #debugging
            if logf != "":
                nCellsNonEmpty = np.count_nonzero(cells.mesh.area)
                DV_len = cells.mesh.geometry.width
                AP_len = cells.mesh.geometry.height
                AP2DV_ratio = AP_len / DV_len
                fout.write("%d\t%d\t%d\t%g\t%g\t%g\t%g\t%g\t%g\t%g\n" % (run_counter/time_step, cells.mesh.n_face, nCellsNonEmpty, DV_len, AP_len, AP2DV_ratio, nT1, nT2, nD, dtime))
            
            ####print division to file###
            if SIM_TYPE == 1:
                write_to_division_extended_file(foutD, run_counter/time_step, nD, 0, properties['time_stamp_division'], properties['edge_division'])
            else:
                write_to_division_extended_file(foutD, run_counter/time_step, 0, nD, properties['time_stamp_division'], properties['edge_division'])
            
            ####print lineage to file### (ids of divisions and ids of daughter cells)
            if SIM_TYPE == 1:
                write_to_division_lineage_file(foutDL, run_counter/time_step, nD, 0, properties['ids_division'], properties['ids_daughters'])
            else:
                write_to_division_lineage_file(foutDL, run_counter/time_step, 0, nD, properties['ids_division'], properties['ids_daughters'])
			
			####print T1 and T2 ids to file### (ids of divisions and ids of daughter cells)
            #write_to_T1_cell_ids_file(foutDT, run_counter/time_step, nT1, nT2, properties['T1_ids'], properties['T2_ids'])
            write_to_T1_dict_file(foutDT, run_counter/time_step, T1_dict)
                        
            ####print T1 and T2 to file###
            write_to_T1_extended_file(foutT1, run_counter/time_step, nT1, nT2, properties['time_stamp_T1'], properties['time_stamp_remove'], properties['mid_point_T1'], properties['edge_double_remove'])
            
            ####reset counters####
            T1_CPUtimeA = 0
            T1_CPUtimeB = 0
            nT1 = 0
            nT2 = 0
            nD = 0
            T1_dict = {}
            
            properties['ids_division'] = []
            #properties['Division_angle_pD'] = []
            properties['time_stamp_division'] = []  #MZ division tracking
            properties['time_stamp_T1'] = []  #MZ T1 tracking
            properties['time_stamp_remove'] = []  #MZ remove (T2, double edge) tracking
            properties['edge_division'] = []  #MZ division tracking
            properties['mid_point_T1'] = []  #MZ T1 tracking
            properties['edge_double_remove'] = []  #MZ remove tracking
            properties['ids_daughters'] = []
            properties['T1_ids'] = [] #MZ T1 cell ids
            properties['T2_ids'] = [] #MZ ids of removed cells
        run_counter += 1
        #########Clock End##########
        
        if hasattr(cells.mesh.geometry,'width'):
            expansion[0] = expansion_constant*np.average(F[0]*cells.mesh.vertices[0])*dt/(cells.mesh.geometry.width**2)
        if hasattr(cells.mesh.geometry,'height'): #Cylinder mesh doesn't have 'height' argument
            expansion[1] = np.average(F[1]*cells.mesh.vertices[1])*dt/(cells.mesh.geometry.height**2)
        cells.mesh = cells.mesh.moved(dv).scaled(1.0+expansion)
        yield cells



"""Run simulation and save data functions"""
def run_simulation_INM_screen(x, x0, nX0, nY0, timestart, timend, rand, sim_type, fout, foutD, foutDL, foutDT, foutT1):       #Zagorski: timestart as parameter #fout is for log information
    global dt, time_initial, run_counter, time_unit, life_span, logf, T1_eps, nT1, nT2, nD, T1_dict
    logf = 1 #MZ, print log to file
    time_initial = time.time()  #MZ
    steps_per_unit = int(time_unit / dt) #MZ
    run_counter = 0             #MZ
    T1_dict = {}                #MZ: keys are T1 edge ids, and values numbers of T1 transition events for that edge
    #sim_type 0 simulation_with_division_clone (no differentiation rate)
    #sim_type 1 simulation_with_division_clone_differentiation (all differentiation rate)
    #sim_type 2 simulation_with_division_clone_differenciation_3stripes (2 population with and without diffentiation rate)
        
    #K, G, L for the 1st stage
    K=x0[0]
    G=x0[1]
    L=x0[2]
    L_std_IN = L_std
    L_tau_IN = L_tau
    
    rand1 = rand    #MZ
    # mesh = init.cylindrical_hex_mesh(10,10,noise=0.2,rand=rand1)
    mesh = init.toroidal_hex_mesh(nX0,nY0,noise=0.2,rand=rand1)
    cells = model.Cells(mesh,properties={'K':K,'Gamma':G,'P':0.0,'boundary_P':P,'Lambda':L, 'Lambda_boundary':0.5})
    cells.properties['age'] = rand.rand(len(cells))   #MZ: np.random -> rand
    cells.properties['parent_group'] = np.zeros(len(cells),dtype=int)
    cells.properties['time_stamp_division'] = [] #MZ
    n_edges = len(cells.mesh.face_id_by_edge)       #MZ
    cells.properties['Lambda_edge'] = np.full(n_edges, L)   #MZ
    set_OU_noise(0, 0)               #NO noise in the 1st stage
    get_noise_edges(cells, L_std, rand=rand1)       #MZ fills cells.properties['Lambda_edge'] with random numbers N(L, L_std)
    
    force = TargetArea() + Tension() + Perimeter() + Pressure()
    history1 = run(simulation_with_division_TYPE_noise(0, cells,force,fout,foutD,foutDL,foutDT,foutT1,T1_eps=0.01,lifespan=100,rand=rand1),int(timestart/dt), steps_per_unit) #200/dt, by default 150/dt, modified to timestart/dt for flexibility
    print("1st stage, L_std: %g, L_tau: %g"%(L_std,L_tau))
    cells = history1[-1].copy()
    
    #K, G, L for the 2nd stage
    K=x[0]
    G=x[1]
    L=x[2]
    set_OU_noise(L_tau_IN,L_std_IN)       #
    print("2nd_stage, L_std: %g, L_tau: %g"%(L_std,L_tau))
    
    cells.properties['K'] = K
    cells.properties['Gamma'] = G
    cells.properties['Lambda'] = L
    n_edges = len(cells.mesh.face_id_by_edge)
    cells.properties['Lambda_edge'] = np.full(n_edges, L)
    get_noise_edges(cells, L_std, rand=rand1)       #MZ fills cells.properties['Lambda_edge'] with random numbers N(L, L_std)
    
    cells.properties['parent_group'] = np.zeros(len(cells),dtype=int) #use to draw clone
    cells.properties['parent'] = cells.mesh.face_ids #save the ids to control division parents-daugthers 
    cells.properties['parent_group'] = bin_by_xpos(cells,np.cumsum([0.35,0.3,0.35]))
    
    #reset division, T1 and T2 (cell removal) counters #MZ, added on 1.04.2019
    nD  = 0
    nT1 = 0.
    nT2 = 0.
    T1_dict = {}    #clears dictionary before second stage of simulation
    
    if sim_type == 0:
        print("Second stage")
        #fout.write("Second stage\n")
        history1[-1].properties['parent_group'] = np.zeros(len(history1[-1].properties['parent_group']),dtype=int)
        history = run(simulation_with_division_TYPE_noise(0, cells,force,fout,foutD,foutDL,foutDT,foutT1,T1_eps=T1_eps,lifespan=life_span,rand=rand),int(timend/dt),steps_per_unit)
        #previous to 2019.04.16, simulation_with_division_clone function was used, but it is the same as simulation_with_division in current setup
        #history = run(simulation_with_division_clone(cells,force,rand=rand),(timend)/dt,10.0) #detailed recording!!!!!!!
    if sim_type == 1:
        print("Second stage")
        #fout.write("Second stage\n")
        history1[-1].properties['parent_group'] = np.zeros(len(history1[-1].properties['parent_group']),dtype=int)+1
        history = run(simulation_with_division_TYPE_noise(1, cells,force,fout,foutD,foutDL,foutDT,foutT1,T1_eps=T1_eps,lifespan=life_span,rand=rand),int(timend/dt),steps_per_unit)
        history[-1].properties['parent_group'] = np.zeros(len(history[-1].properties['parent_group']),dtype=int)+1
    
    return history1, history