# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import functools
import copy
import os
import time #MZ: debugging
import numpy as np
import sys

from permutations import cycle

# ====from cells.py====
class Cells(object):
    def __init__(self, mesh, properties=None):
        self.mesh = mesh
        self.properties = properties or {}

    def copy(self):
        # mesh is 'immutable' (so we can cache shared computations) => no need to copy it
        #print(self.properties.copy())
        return Cells(self.mesh, self.properties.copy())

    def __len__(self):
        return self.mesh.n_face

    def empty(self):
        # self.mesh._edge_lookup[self.mesh.area<0]=-1 ###########anyadido por mi!!!!!!!!! 
        return self.mesh._edge_lookup == -1

    def by_face(self, property_name, boundary_property_name=None):
        value = self.properties[property_name]
        if self.mesh.has_boundary():
            value = make_array(value, len(self))
            boundary_value = self.properties[boundary_property_name] if boundary_property_name else 0.0
            value[self.mesh.boundary_faces] = boundary_value
        return value
    
    def by_face_array(self, property_name):      #added to have noise in LAMBDA #MZ, 2020.07.15 #NOT USED when specific conditions on the boundary are present
        value = self.properties[property_name]
        value = make_array(value, len(self))
        return value

    def by_edge(self, property_name, boundary_property_name=None):
        value_by_face = self.by_face(property_name, boundary_property_name)
        if not hasattr(value_by_face, 'shape'):  # scalar
            return value_by_face
        return value_by_face.take(self.mesh.face_id_by_edge)
    
    def by_edge_removal(self, property_name, boundary_property_name=None): #correct removal of differentiating cells for negative Lambda #MZ, 2019.10.02
        value_by_face = self.by_face(property_name, boundary_property_name)
        #print(value_by_face)
        value_by_face_A0 = self.by_face('A0')
        keep_tab = (value_by_face_A0 > 0.).astype(int)     #returns 1 for cells with non-zero target area and 0 otherwise
        removal_tab = (value_by_face_A0 == 0.).astype(int) #returns 1 for cells with zero target area and 0 otherwise
        value_by_face = value_by_face*keep_tab + 0.2*removal_tab       #results in 0.2 Lambda for edges in cells with A0 = 0
        if not hasattr(value_by_face, 'shape'):  # scalar
            return value_by_face
        return value_by_face.take(self.mesh.face_id_by_edge)
    
    def by_edge_removal_Lambda_noise(self, property_name): #correct removal of differentiating cells for negative Lambda #MZ, 2019.10.02; #Lambda noise 2020.07.20
        value_by_face_array = self.by_face_array(property_name)
        
        ####remove empty cells by setting positive Lambda (=0.2 by default) on the edges of cell-to-be-removed###
        value_by_face_A0 = self.by_face('A0')
        value_by_face_A0_array = value_by_face_A0.take(self.mesh.face_id_by_edge)       #MZ, A0 area of face associated with the edge
        keep_tab = (value_by_face_A0_array > 0.).astype(int)     #returns 1 for cells with non-zero target area and 0 otherwise
        removal_tab = (value_by_face_A0_array <= 0.).astype(int) #returns 1 for cells with zero target area and 0 otherwise
        value_by_face_array = value_by_face_array*keep_tab + 0.2*removal_tab       #results in 0.2 Lambda for edges in cells with A0 = 0
        ####end of setting positive Lambda for cell removal####
        
        return value_by_face_array              #MZ no need to select lambda for every edge for a given face, as it already has the correct format

def make_array(value, n):
    if hasattr(value, 'shape'):
        return value
    expanded = np.empty(n, type(value))
    expanded.fill(value)
    return expanded

# ====from mesh.py====
# ====Cells data structure====
def cached_property(func):
    """A decorator for caching class properties."""
    @functools.wraps(func)
    def getx(self):
        try:
            return self._cache[func]
        except AttributeError:
            self._cache = {}
        except KeyError:
            pass
        self._cache[func] = res = func(self)
        return res
    return property(getx)


_MAX_CELLS = 10**5


class Edges(object):
    """
    Attributes:
        ids: The array of integers [0,1,...,n_edge-1] giving the ID of each (directed) edge.
        rotate: An array of n_edge integers giving the ID of the next edge around a vertex.
        rotate2: An array of n_edge integers giving the ID of the previous edge around a vertex.
        reverse: An array of n_edge integers giving the ID of the reversed edge.
    """
    # share fixed arrays between objects for efficiency
    IDS = np.arange(6*_MAX_CELLS)
    ROTATE = np.roll(IDS.reshape(-1, 3), 2, axis=1).ravel()
    ROTATE2 = ROTATE[ROTATE]

    def __init__(self, reverse):
        self.reverse = reverse
        n = len(reverse)
        self.rotate = self.ROTATE[:n]
        self.rotate2 = self.ROTATE2[:n]
        self.ids = self.IDS[:n]

        # make 'immutable' so that various computed properties can be cached
        reverse.setflags(write=False)

    def __len__(self):
        return len(self.reverse)

    @cached_property
    def next(self):
        """An array of n_edge integers giving the ID of the next edge around the (left) face."""
        return self.rotate[self.reverse]

    @cached_property
    def prev(self):
        """An array of n_edge integers giving the ID of the previous edge around the (left) face."""
        return self.reverse[self.rotate2]


class MeshGeometry(object):
    def scaled(self, scale):
        return self

    def recentre(self, mesh):
        return mesh


class Plane(MeshGeometry):
    pass


class Torus(MeshGeometry):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def scaled(self, scale):
        return Torus(self.width*scale[0], self.height*scale[1])

    def recentre(self, mesh):
        counts = mesh.edge_counts()[mesh.face_id_by_edge]
        for i, L in enumerate((self.width, self.height)):
            mid = np.bincount(mesh.face_id_by_edge, mesh.vertices[i])[mesh.face_id_by_edge]
            mesh.vertices[i, mid > 0.5*L*counts] -= L
            mesh.vertices[i, mid < -0.5*L*counts] += L
        return mesh


class Cylinder(MeshGeometry):
    def __init__(self, width):
        self.width = width

    def scaled(self, scale):
        return Cylinder(self.width*scale[0])

    def recentre(self, mesh):
        counts = mesh.edge_counts()[mesh.face_id_by_edge]
        L = self.width
        mid = np.bincount(mesh.face_id_by_edge, mesh.vertices[0])[mesh.face_id_by_edge]
        mesh.vertices[0, mid > 0.5*L*counts] -= L
        mesh.vertices[0, mid < -0.5*L*counts] += L
        return mesh


class Mesh(object):
    """
    An mesh in the 'half-edge' representation.

    meshes are intended to be immutable so that we can cache various computed properties (such as
    lengths of edges) for efficiency. Methods to move vertices, or change the topology all return new
    Meshs object, leaving the original intact.

    Args:
        edges: An Edges object holding topological information about the network of cells.
        vertices: A (2,n_edge) array of floats giving the x,y position of the start vertex of each edge.
        face_ids: An (n_edge) array of integers holding the face_id for each edge
        geometry: An instance of MeshGeometry, holding information about the background geometry
        n_face: total number of cells including 'dead' cells which are no longer in edges network.
    """

    def __init__(self, edges, vertices, face_id_by_edge, geometry, n_face=None, boundary_faces=None):
        self.edges = edges
        self.vertices = vertices
        self.face_id_by_edge = face_id_by_edge
        face_id_by_edge.setflags(write=False)
        self.n_face = n_face or np.max(face_id_by_edge) + 1
        self.geometry = geometry
        self.boundary_faces = boundary_faces
        
    def copy(self):
        mesh = copy.copy(self)
        mesh._cache = {}
        return mesh

    def scaled(self, scale):
        mesh = self.copy()
        mesh.vertices = mesh.vertices * scale[:, None]
        mesh.geometry = mesh.geometry.scaled(scale)
        return mesh

    def recentre(self):
        mesh = self.copy()
        return mesh.geometry.recentre(mesh)

    def has_boundary(self):
        return self.boundary_faces is not None

    @cached_property
    def boundary_edges(self):
        edges = np.hstack([self.boundary(face) for face in self.boundary_faces])
        return edges

    @cached_property
    def face_ids(self):
        return np.arange(self.n_face)

    @cached_property
    def edge_vect(self):
        v = self.vertices
        dv = v.take(self.edges.next, 1) - v
        if self.has_boundary():
            if np.any(self.boundary_edges==-1):
                # print self.boundary_edges
                os._exit(1)
            dv[:, self.boundary_edges] = -dv.take(self.edges.reverse[self.boundary_edges], 1)
        return dv

    @cached_property
    def length(self):
        vect = self.edge_vect
        return np.sqrt(vect[0]**2 + vect[1]**2)

    @cached_property
    def edge_angle(self):
        vect = self.edge_vect
        angle = np.arctan(vect[1]/vect[0])
        return angle

    @cached_property
    def area(self):
        return np.bincount(self.face_id_by_edge, 0.5*np.cross(self.vertices.T, self.edge_vect.T), self.n_face)

    @cached_property
    def perimeter(self):
        return np.bincount(self.face_id_by_edge, self.length, self.n_face)

    @cached_property
    def d_area(self):
        dv = self.edge_vect
        dA = np.empty_like(dv)
        dA[0], dA[1] = dv[1], -dv[0]
        return 0.5*(dA + dA.take(self.edges.prev, 1))

    @cached_property
    def d_perimeter(self):
        dL = self.edge_vect/self.length
        return dL - dL.take(self.edges.prev, 1)

    def edge_counts(self):
        return np.bincount(self.face_id_by_edge)

    @cached_property
    def _edge_lookup(self):
        lookup = np.empty(self.n_face, int)
        lookup.fill(-1)
        lookup[self.face_id_by_edge] = self.edges.ids
        return lookup

    def boundary(self, cell_id):
        edge = self._edge_lookup[cell_id]
        if edge == -1:
            return [-1]
            os._exit(1) #force end program because edge is out of the edges ids #Pilar have added it!!!
        if edge != -1: 
            return cycle(self.edges.next, edge)

    def moved(self, dv):
        """Caller is responsible for making dv consistent for different copies of a vertex."""
        mesh = self.copy()
        mesh.vertices = mesh.vertices + dv
        return mesh

    def transition(self, eps):
        return _transition(self, eps)
    
    def transition_count(self, eps):
        return _transition_count(self, eps)
    
    def transition_count_more(self, eps):
        return _transition_count_more(self, eps)
    
    def transition_count_extended(self, eps):
        return _transition_count_extended(self, eps)
    
    def transition_count_extended_track(self, eps):
        return _transition_count_extended_track(self, eps)
    
    def transition_count_T1_dict(self, eps):
        return _transition_count_T1_dict(self, eps)
    
    def transition_count_T1_dict_del(self, eps):
        return _transition_count_T1_dict_del(self, eps)

    def add_edges(self, edge_pairs):
        return _add_edges(self, edge_pairs)
    
    def add_edges_extended(self, edge_pairs):
        return _add_edges_extended(self, edge_pairs)


def _remove(edges, reverse, vertices, face_id_by_edge):
    """Removes the given edges and their orbit under rotation (ie removes complete vertices.)

    Args:
        edges: An array of edge IDs to remove
        reverse: see mesh.edges.reverse
        vertices: see mesh.vertices
        face+ids: see mesh.face_ids

    Returns:
        A tuple of reverse,vertex,cell arrays with the edges removed.
    """
    es = np.unique(edges//3*3)
    to_del = np.dstack([es, es+1, es+2]).ravel()  # orbit of the given edges under rotation
    reverse = np.delete(reverse, to_del)
    reverse = (np.cumsum(np.bincount(reverse))-1)[reverse]  # relabel to get a perm of [0,...,N-1]
    #print(es)
    #print("to_del",to_del)
    #print(vertices[:,to_del])
    vertices = np.delete(vertices, to_del, 1)
    face_id_by_edge = np.delete(face_id_by_edge, to_del)
    return reverse, vertices, face_id_by_edge

def _remove_get_del(edges, reverse, vertices, face_id_by_edge):
    """Removes the given edges and their orbit under rotation (ie removes complete vertices.)

    Args:
        edges: An array of edge IDs to remove
        reverse: see mesh.edges.reverse
        vertices: see mesh.vertices
        face+ids: see mesh.face_ids

    Returns:
        A tuple of reverse,vertex,cell arrays with the edges removed.
    """
    es = np.unique(edges//3*3)
    to_del = np.dstack([es, es+1, es+2]).ravel()  # orbit of the given edges under rotation
    reverse = np.delete(reverse, to_del)
    reverse = (np.cumsum(np.bincount(reverse))-1)[reverse]  # relabel to get a perm of [0,...,N-1]
    #print(es)
    #print("to_del",to_del)
    #print(vertices[:,to_del])
    vertices = np.delete(vertices, to_del, 1)
    face_id_by_edge = np.delete(face_id_by_edge, to_del)
    return reverse, vertices, face_id_by_edge, to_del       #MZ: returns "to_del" to keep track of removed edges and remove assigned Lambda values


# def sum_vertices(edges,dv):
#    return np.repeat(dv[:,::3]+dv[:,1::3]+dv[:,2::3],3,1)
def sum_vertices(edges, dv):
    return dv + np.take(dv, edges.rotate, 1) + np.take(dv, edges.rotate2, 1)


def _T1(edge, eps, rotate, reverse, vertices, face_id_by_edge):
    global T1_newf
    T1_newf=1.01
    e0 = edge
    e1 = rotate[edge]
    e2 = rotate[e1]
    e3 = reverse[edge]
    e4 = rotate[e3]
    e5 = rotate[e4]

    before = np.array([e1, e2, e4, e5])
    after = np.array([e2, e4, e5, e1])

    after_r = reverse.take(after)
    reverse[before] = after_r
    reverse[after_r] = before

    dv = vertices[:, e4]-vertices[:, e0]
    l = T1_newf*eps/np.sqrt(dv[0]*dv[0]+dv[1]*dv[1]) 
    dw = [dv[1]*l, -dv[0]*l]

    for i in [0, 1]:
        dp = 0.5*(dv[i]+dw[i])
        dq = 0.5*(dv[i]-dw[i])
        v = vertices[i]
        v[before] = v.take(after) + np.array([dp, -dq, -dp, dq])
        v[e0] = v[e4] + dw[i]
        v[e3] = v[e1] - dw[i]

    face_id_by_edge[before] = face_id_by_edge.take(after)
    face_id_by_edge[e0] = face_id_by_edge[e4]
    face_id_by_edge[e3] = face_id_by_edge[e1]

    return np.hstack([before, after_r])

def _T1_extended(edge, eps, rotate, reverse, vertices, face_id_by_edge):
    global T1_newf
    T1_newf=1.01
    e0 = edge
    e1 = rotate[edge]
    e2 = rotate[e1]
    e3 = reverse[edge]
    e4 = rotate[e3]
    e5 = rotate[e4]
    
    T1_cell_ids = [face_id_by_edge[e0], face_id_by_edge[e3]]    #(A, B, C, D), 4 cell ids before T1 transition; A neighbours B; after T1: C neighbours D
    
    before = np.array([e1, e2, e4, e5])
    after = np.array([e2, e4, e5, e1])

    after_r = reverse.take(after)
    reverse[before] = after_r
    reverse[after_r] = before

    dv = vertices[:, e4]-vertices[:, e0]
    l = T1_newf*eps/np.sqrt(dv[0]*dv[0]+dv[1]*dv[1])
    dw = [dv[1]*l, -dv[0]*l]

    for i in [0, 1]:
        dp = 0.5*(dv[i]+dw[i])
        dq = 0.5*(dv[i]-dw[i])
        v = vertices[i]
        v[before] = v.take(after) + np.array([dp, -dq, -dp, dq])
        v[e0] = v[e4] + dw[i]
        v[e3] = v[e1] - dw[i]

    face_id_by_edge[before] = face_id_by_edge.take(after)
    face_id_by_edge[e0] = face_id_by_edge[e4]
    face_id_by_edge[e3] = face_id_by_edge[e1]
    
    T1_cell_ids = T1_cell_ids + [face_id_by_edge[e0], face_id_by_edge[e3]] #(A, B, C, D), 4 cell ids before T1 transition; A neighbours B; after T1: C neighbours D
   
    return np.hstack([before, after_r]), T1_cell_ids


def _transition(mesh, eps):                   #MZ: use _transiiton_count to count T1 and T2 transitions
    edges = mesh.edges
    half_edges = edges.ids[edges.ids < edges.reverse]  # half the edges have id < reverse_id
    dv = mesh.edge_vect.take(half_edges, 1)
    short_edges = set(half_edges[np.sum(dv*dv, 0) < eps*eps])
    ids_t1=half_edges[np.sum(dv*dv, 0) < eps*eps]
    
    if not short_edges:
        return mesh, ids_t1
    reverse, vertices, face_id_by_edge = edges.reverse.copy(), mesh.vertices.copy(), mesh.face_id_by_edge.copy()
    rotate = edges.rotate
    # Do T1 transitions
    # to avoid nasty edge cases, we don't allow T1's to happen on adjacent edges
    # and delay to the next timestep if necessary.
    # A better approach would be to take multiple partial timesteps.
    boundary_edges = mesh.boundary_edges if mesh.has_boundary() else []
    while short_edges:
        edge = short_edges.pop()
        if edge in boundary_edges:
            edge = reverse[edge]
        neighbours = _T1(edge, eps, rotate, reverse, vertices, face_id_by_edge)
        for x in neighbours:
            short_edges.discard(x)      

    # Remove collapsed (ie two-sided) faces.
    while True:
        nxt = rotate[reverse]
        two_sided = np.where(nxt[nxt] == edges.ids[:len(nxt)])[0]
        if not len(two_sided):
            break
        while np.any(reverse[reverse[rotate[two_sided]]] != reverse[rotate[nxt[two_sided]]]):
            reverse[reverse[rotate[two_sided]]] = reverse[rotate[nxt[two_sided]]]
        prev_face_id_by_edge = face_id_by_edge
        reverse, vertices, face_id_by_edge = _remove(two_sided, reverse, vertices, face_id_by_edge)
        ids_removed = np.setdiff1d(prev_face_id_by_edge,face_id_by_edge)
        #if ~(ids_t1==np.delete(ids_t1,ids_removed)):
        #    print 'Ids T1 to remove:', ids_t1, ids_removed, np.delete(ids_t1,ids_removed)

    mesh = mesh.copy()
    mesh.edges = Edges(reverse)
    mesh.vertices = vertices
    mesh.face_id_by_edge = face_id_by_edge
    return mesh, ids_t1

def _transition_count_more(mesh, eps):          #function for listing more information about t1 and t2; significant slow down for oscillatory t1 transitions; not fully tested
    T1_counter = 0
    T2_counter = 0
    edges = mesh.edges
    half_edges = edges.ids[edges.ids < edges.reverse]  # half the edges have id < reverse_id
    dv = mesh.edge_vect.take(half_edges, 1)
    short_edges = set(half_edges[np.sum(dv*dv, 0) < eps*eps])
    ids_t1=half_edges[np.sum(dv*dv, 0) < eps*eps]
    
    #MZ debugging
    half_edges_up = edges.ids[edges.ids > edges.reverse]  # half the edges have id < reverse_id
    dv_up = mesh.edge_vect.take(half_edges_up, 1)
    short_edges_up = set(half_edges_up[np.sum(dv_up*dv_up, 0) < eps*eps])
    ids_t1_up=half_edges_up[np.sum(dv_up*dv_up, 0) < eps*eps]
    #end debugging
    two_sided_ver = []
    
    #if len(ids_t1):
    #    print 'Ids T1 to T1:', ids_t1, mesh.face_id_by_edge[ids_t1]
    #    print 'T1_edge vec:', mesh.edge_vect.T[ids_t1]
    #    print 'T1_edge ver:', mesh.vertices.T[ids_t1]
    #    print 'T1_mid_point:', mesh.vertices.T[ids_t1] + 0.5*mesh.edge_vect.T[ids_t1]
    #    print 'Ids T1 to T1 (UP):', ids_t1_up, mesh.face_id_by_edge[ids_t1_up]
    #    print 'T1_edge vec (UP):', mesh.edge_vect.T[ids_t1_up]
    #    print 'T1_edge ver (UP):', mesh.vertices.T[ids_t1_up]
    
    if not short_edges:
        return mesh, ids_t1, T1_counter, T2_counter
    reverse, vertices, face_id_by_edge = edges.reverse.copy(), mesh.vertices.copy(), mesh.face_id_by_edge.copy()
    rotate = edges.rotate
    # Do T1 transitions
    # to avoid nasty edge cases, we don't allow T1's to happen on adjacent edges
    # and delay to the next timestep if necessary.
    # A better approach would be to take multiple partial timesteps.
    boundary_edges = mesh.boundary_edges if mesh.has_boundary() else []
    #print(short_edges)
    while short_edges:
        edge = short_edges.pop()
        if edge in boundary_edges:
            edge = reverse[edge]
        neighbours = _T1(edge, eps, rotate, reverse, vertices, face_id_by_edge)
        T1_counter = T1_counter + 1
        for x in neighbours:
            short_edges.discard(x)      

    # Remove collapsed (ie two-sided) faces.
    while True:
        nxt = rotate[reverse]
        two_sided = np.where(nxt[nxt] == edges.ids[:len(nxt)])[0]
        stop_loop = 0 #MZ; debugging; escapes infinite loop when mesh colapses
        if not len(two_sided):
            break
        while np.any(reverse[reverse[rotate[two_sided]]] != reverse[rotate[nxt[two_sided]]]):
            reverse[reverse[rotate[two_sided]]] = reverse[rotate[nxt[two_sided]]]
            stop_loop = stop_loop + 1   #MZ; debugging; escapes infinite loop when mesh colapses
            if stop_loop == 1000:
                print("Cannot perform T1 transitions. Mesh failed. Simulation is stopped.")
                sys.exit(-1)        #MZ; !!!EXIT when mesh colapses
        prev_face_id_by_edge = face_id_by_edge
        reverse, vertices, face_id_by_edge = _remove(two_sided, reverse, vertices, face_id_by_edge)
        ids_removed = np.setdiff1d(prev_face_id_by_edge,face_id_by_edge)
        #T2_counter = T2_counter + 1   #underestimates when more than 1 two_sided are in a single time step (very rare)
        T2_counter = T2_counter + len(two_sided)/2
        #print("T2_counter", T2_counter)
        #if ~(ids_t1==np.delete(ids_t1,ids_removed)):
        #print 'Ids T1 to remove:', ids_t1, ids_removed, np.delete(ids_t1,ids_removed)
        #print 'Ids T1 to remove:', ids_t1, two_sided, ids_removed
        #print 'two_sided vec:', mesh.edge_vect.T[two_sided]
        #print 'two_sided ver:', mesh.vertices.T[two_sided]
        two_sided_ver = np.append(two_sided_ver, mesh.vertices.T[two_sided])

    #print 'two sided tab:',two_sided_ver
    mesh = mesh.copy()
    mesh.edges = Edges(reverse)
    mesh.vertices = vertices
    mesh.face_id_by_edge = face_id_by_edge
    return mesh, ids_t1, T1_counter, T2_counter#, mesh.face_id_by_edge[ids_t1], mesh.face_id_by_edge[reverse[ids_t1]], ids_removed

def _transition_count_extended(mesh, eps):
    T1_counter = 0
    T2_counter = 0
    edges = mesh.edges
    half_edges = edges.ids[edges.ids < edges.reverse]  # half the edges have id < reverse_id
    dv = mesh.edge_vect.take(half_edges, 1)
    short_edges = set(half_edges[np.sum(dv*dv, 0) < eps*eps])
    ids_t1=half_edges[np.sum(dv*dv, 0) < eps*eps]
    ids_t1_done = []  #does not take into account T1s on adjacent edges
    
    two_sided_ver = []
    T1_cell_all = []
    ids_removed = []
    mid_point_T1 = mesh.vertices.T[ids_t1] + 0.5*mesh.edge_vect.T[ids_t1]
    
    #if len(ids_t1):
    #    print 'T1_mid_point:', mid_point_T1
    
    if not short_edges:
        return mesh, ids_t1, T1_counter, T2_counter, mid_point_T1, two_sided_ver, T1_cell_all, ids_removed
    reverse, vertices, face_id_by_edge = edges.reverse.copy(), mesh.vertices.copy(), mesh.face_id_by_edge.copy()
    rotate = edges.rotate
    # Do T1 transitions
    # to avoid nasty edge cases, we don't allow T1's to happen on adjacent edges
    # and delay to the next timestep if necessary.
    # A better approach would be to take multiple partial timesteps.
    boundary_edges = mesh.boundary_edges if mesh.has_boundary() else []
    #print(short_edges)
    while short_edges:
        edge = short_edges.pop()
        ids_t1_done.append(edge)
        if edge in boundary_edges:
            edge = reverse[edge]
        neighbours, T1_cell_ids = _T1_extended(edge, eps, rotate, reverse, vertices, face_id_by_edge)
        T1_cell_all = np.append(T1_cell_all, T1_cell_ids)
        T1_counter = T1_counter + 1
        for x in neighbours:
            short_edges.discard(x)      
    
    mid_point_T1_done = mesh.vertices.T[ids_t1_done] + 0.5*mesh.edge_vect.T[ids_t1_done]   
            
    # Remove collapsed (ie two-sided) faces.
    while True:
        nxt = rotate[reverse]
        two_sided = np.where(nxt[nxt] == edges.ids[:len(nxt)])[0]
        stop_loop = 0 #MZ; debugging; escapes infinite loop when mesh colapses
        if not len(two_sided):
            break
        while np.any(reverse[reverse[rotate[two_sided]]] != reverse[rotate[nxt[two_sided]]]):
            reverse[reverse[rotate[two_sided]]] = reverse[rotate[nxt[two_sided]]]
            stop_loop = stop_loop + 1   #MZ; debugging; escapes infinite loop when mesh colapses
            if stop_loop == 1000:
                print("Cannot perform T1 transitions. Mesh failed. Simulation is stopped.")
                sys.exit(-1)        #MZ; !!!EXIT when mesh colapses
        prev_face_id_by_edge = face_id_by_edge
        two_sided_ver = np.append(two_sided_ver, mesh.vertices.T[two_sided])  #vertices of two-sided edge that was removed
        reverse, vertices, face_id_by_edge = _remove(two_sided, reverse, vertices, face_id_by_edge)
        ids_removed = np.setdiff1d(prev_face_id_by_edge,face_id_by_edge)
        #T2_counter = T2_counter + 1   #underestimates when more than 1 two_sided are in a single time step (very rare)
        T2_counter = T2_counter + len(two_sided)/2
        #print 'Two sided ver:', two_sided_ver
        #print("T2_counter", T2_counter)
        #if ~(ids_t1==np.delete(ids_t1,ids_removed)):
        #print('Ids T2 to remove:', ids_removed)
    
    mesh = mesh.copy()
    mesh.edges = Edges(reverse)
    mesh.vertices = vertices
    mesh.face_id_by_edge = face_id_by_edge
    return mesh, ids_t1_done, T1_counter, T2_counter, mid_point_T1_done, two_sided_ver, T1_cell_all, ids_removed

def _transition_count_extended_track(mesh, eps):
    T1_counter = 0
    T2_counter = 0
    edges = mesh.edges
    half_edges = edges.ids[edges.ids < edges.reverse]  # half the edges have id < reverse_id
    dv = mesh.edge_vect.take(half_edges, 1)
    short_edges = set(half_edges[np.sum(dv*dv, 0) < eps*eps])
    ids_t1=half_edges[np.sum(dv*dv, 0) < eps*eps]
    ids_t1_done = []  #does not take into account T1s on adjacent edges
    
    two_sided_ver = []
    T1_cell_all = []
    T1_edge = np.empty(0,dtype=int)     #both [e0, e3] edges are used to identify T1; e3 = reverse[e0]
    ids_removed = []
    mid_point_T1 = mesh.vertices.T[ids_t1] + 0.5*mesh.edge_vect.T[ids_t1]
    
    #if len(ids_t1):
    #    print 'T1_mid_point:', mid_point_T1
    
    if not short_edges:
        return mesh, ids_t1, T1_counter, T2_counter, mid_point_T1, two_sided_ver, T1_cell_all, ids_removed, T1_edge
    reverse, vertices, face_id_by_edge = edges.reverse.copy(), mesh.vertices.copy(), mesh.face_id_by_edge.copy()
    rotate = edges.rotate
    # Do T1 transitions
    # to avoid nasty edge cases, we don't allow T1's to happen on adjacent edges
    # and delay to the next timestep if necessary.
    # A better approach would be to take multiple partial timesteps.
    boundary_edges = mesh.boundary_edges if mesh.has_boundary() else []
    #print(short_edges)
    while short_edges:
        edge = short_edges.pop()
        ids_t1_done.append(edge)
        if edge in boundary_edges:
            edge = reverse[edge]
        neighbours, T1_cell_ids = _T1_extended(edge, eps, rotate, reverse, vertices, face_id_by_edge)
        T1_cell_all = np.append(T1_cell_all, T1_cell_ids)
        T1_counter = T1_counter + 1
        T1_edge = np.append(T1_edge, [edge, reverse[edge]]) #T1_edge is flattened 
        #print("T1 edges: ", edge, reverse[edge])
        
        for x in neighbours:
            short_edges.discard(x)      
    
    mid_point_T1_done = mesh.vertices.T[ids_t1_done] + 0.5*mesh.edge_vect.T[ids_t1_done]   
            
    # Remove collapsed (ie two-sided) faces.
    while True:
        nxt = rotate[reverse]
        two_sided = np.where(nxt[nxt] == edges.ids[:len(nxt)])[0]
        stop_loop = 0 #MZ; debugging; escapes infinite loop when mesh colapses
        if not len(two_sided):
            break
        while np.any(reverse[reverse[rotate[two_sided]]] != reverse[rotate[nxt[two_sided]]]):
            reverse[reverse[rotate[two_sided]]] = reverse[rotate[nxt[two_sided]]]
            stop_loop = stop_loop + 1   #MZ; debugging; escapes infinite loop when mesh colapses
            if stop_loop == 1000:
                print("Cannot perform T1 transitions. Mesh failed. Simulation is stopped.")
                sys.exit(-1)        #MZ; !!!EXIT when mesh colapses
        prev_face_id_by_edge = face_id_by_edge
        two_sided_ver = np.append(two_sided_ver, mesh.vertices.T[two_sided])  #vertices of two-sided edge that was removed
        reverse, vertices, face_id_by_edge = _remove(two_sided, reverse, vertices, face_id_by_edge)
        ids_removed = np.setdiff1d(prev_face_id_by_edge,face_id_by_edge)
        #T2_counter = T2_counter + 1   #underestimates when more than 1 two_sided are in a single time step (very rare)
        T2_counter = T2_counter + len(two_sided)/2
        #print 'Two sided ver:', two_sided_ver
        #print("T2_counter", T2_counter)
        #if ~(ids_t1==np.delete(ids_t1,ids_removed)):
        #print('Ids T2 to remove:', ids_removed)
    
    mesh = mesh.copy()
    mesh.edges = Edges(reverse)
    mesh.vertices = vertices
    mesh.face_id_by_edge = face_id_by_edge
    return mesh, ids_t1_done, T1_counter, T2_counter, mid_point_T1_done, two_sided_ver, T1_cell_all, ids_removed, T1_edge

def _transition_count_T1_dict(mesh, eps):
    T1_counter = 0
    T2_counter = 0
    edges = mesh.edges
    half_edges = edges.ids[edges.ids < edges.reverse]  # half the edges have id < reverse_id
    dv = mesh.edge_vect.take(half_edges, 1)
    short_edges = set(half_edges[np.sum(dv*dv, 0) < eps*eps])
    ids_t1=half_edges[np.sum(dv*dv, 0) < eps*eps]
    ids_t1_done = []  #does not take into account T1s on adjacent edges
    
    T1_edge = np.empty(0,dtype=int)     #both [e0, e3] edges are used to identify T1; e3 = reverse[e0]
    
    if not short_edges:
        return mesh, ids_t1, T1_counter, T2_counter, T1_edge
    reverse, vertices, face_id_by_edge = edges.reverse.copy(), mesh.vertices.copy(), mesh.face_id_by_edge.copy()
    rotate = edges.rotate
    # Do T1 transitions
    # to avoid nasty edge cases, we don't allow T1's to happen on adjacent edges
    # and delay to the next timestep if necessary.
    # A better approach would be to take multiple partial timesteps.
    boundary_edges = mesh.boundary_edges if mesh.has_boundary() else []
    #print(short_edges)
    while short_edges:
        edge = short_edges.pop()
        ids_t1_done.append(edge)
        if edge in boundary_edges:
            edge = reverse[edge]
        neighbours, T1_cell_ids = _T1_extended(edge, eps, rotate, reverse, vertices, face_id_by_edge)
        T1_counter = T1_counter + 1
        T1_edge = np.append(T1_edge, [edge, reverse[edge]]) #T1_edge is flattened 
        #print("T1 edges: ", edge, reverse[edge])
        
        for x in neighbours:
            short_edges.discard(x)   
            
    # Remove collapsed (ie two-sided) faces.
    while True:
        nxt = rotate[reverse]
        two_sided = np.where(nxt[nxt] == edges.ids[:len(nxt)])[0]
        stop_loop = 0 #MZ; debugging; escapes infinite loop when mesh colapses
        if not len(two_sided):
            break
        while np.any(reverse[reverse[rotate[two_sided]]] != reverse[rotate[nxt[two_sided]]]):
            reverse[reverse[rotate[two_sided]]] = reverse[rotate[nxt[two_sided]]]
            stop_loop = stop_loop + 1   #MZ; debugging; escapes infinite loop when mesh colapses
            if stop_loop == 1000:
                print("Cannot perform T1 transitions. Mesh failed. Simulation is stopped.")
                sys.exit(-1)        #MZ; !!!EXIT when mesh colapses
        reverse, vertices, face_id_by_edge = _remove(two_sided, reverse, vertices, face_id_by_edge)
        #T2_counter = T2_counter + 1   #underestimates when more than 1 two_sided are in a single time step (very rare)
        T2_counter = T2_counter + len(two_sided)/2
        #print 'Two sided ver:', two_sided_ver
        #print("T2_counter", T2_counter)
        #if ~(ids_t1==np.delete(ids_t1,ids_removed)):
        #print('Ids T2 to remove:', ids_removed)
    
    mesh = mesh.copy()
    mesh.edges = Edges(reverse)
    mesh.vertices = vertices
    mesh.face_id_by_edge = face_id_by_edge
    return mesh, ids_t1_done, T1_counter, T2_counter, T1_edge

def _transition_count_T1_dict_del(mesh, eps):
    T1_counter = 0
    T2_counter = 0
    edges = mesh.edges
    half_edges = edges.ids[edges.ids < edges.reverse]  # half the edges have id < reverse_id
    dv = mesh.edge_vect.take(half_edges, 1)
    short_edges = set(half_edges[np.sum(dv*dv, 0) < eps*eps])
    ids_t1=half_edges[np.sum(dv*dv, 0) < eps*eps]
    ids_t1_done = []  #does not take into account T1s on adjacent edges
    to_del = []
    
    T1_edge = np.empty(0,dtype=int)     #both [e0, e3] edges are used to identify T1; e3 = reverse[e0]
        
    if not short_edges:
        return mesh, ids_t1, T1_counter, T2_counter, T1_edge, to_del
    
    reverse, vertices, face_id_by_edge = edges.reverse.copy(), mesh.vertices.copy(), mesh.face_id_by_edge.copy()
    
    rotate = edges.rotate
    # Do T1 transitions
    # to avoid nasty edge cases, we don't allow T1's to happen on adjacent edges
    # and delay to the next timestep if necessary.
    # A better approach would be to take multiple partial timesteps.
    boundary_edges = mesh.boundary_edges if mesh.has_boundary() else []
    #print(short_edges)
    while short_edges:
        edge = short_edges.pop()
        ids_t1_done.append(edge)
        if edge in boundary_edges:
            edge = reverse[edge]
        neighbours, T1_cell_ids = _T1_extended(edge, eps, rotate, reverse, vertices, face_id_by_edge)
        T1_counter = T1_counter + 1
        T1_edge = np.append(T1_edge, [edge, reverse[edge]]) #T1_edge is flattened 
        #print("T1 edges: ", edge, reverse[edge])
        
        for x in neighbours:
            short_edges.discard(x)   
            
    # Remove collapsed (ie two-sided) faces.
    
    to_keep0 = np.arange(len(mesh.face_id_by_edge))
    to_keep = to_keep0
    while True:
        nxt = rotate[reverse]
        two_sided = np.where(nxt[nxt] == edges.ids[:len(nxt)])[0]
        stop_loop = 0 #MZ; debugging; escapes infinite loop when mesh colapses
        if not len(two_sided):
            break
        while np.any(reverse[reverse[rotate[two_sided]]] != reverse[rotate[nxt[two_sided]]]):
            reverse[reverse[rotate[two_sided]]] = reverse[rotate[nxt[two_sided]]]
            stop_loop = stop_loop + 1   #MZ; debugging; escapes infinite loop when mesh colapses
            if stop_loop == 1000:
                print("Cannot perform T1 transitions. Mesh failed. Simulation is stopped.")
                sys.exit(-1)        #MZ; !!!EXIT when mesh colapses
        reverse, vertices, face_id_by_edge, to_del = _remove_get_del(two_sided, reverse, vertices, face_id_by_edge)
        to_keep = np.delete(to_keep, to_del)
        #T2_counter = T2_counter + 1   #underestimates when more than 1 two_sided are in a single time step (very rare)
        T2_counter = T2_counter + len(two_sided)/2
        #print 'Two sided ver:', two_sided_ver
        #print("T2_counter", T2_counter)
        #if ~(ids_t1==np.delete(ids_t1,ids_removed)):
        #print('Ids T2 to remove:', ids_removed)
    
    to_del = np.setdiff1d(to_keep0, to_keep, assume_unique = True)
    mesh = mesh.copy()
    mesh.edges = Edges(reverse)
    mesh.vertices = vertices
    mesh.face_id_by_edge = face_id_by_edge
    return mesh, ids_t1_done, T1_counter, T2_counter, T1_edge, to_del

def _transition_count(mesh, eps):
    T1_counter = 0
    T2_counter = 0
    edges = mesh.edges
    half_edges = edges.ids[edges.ids < edges.reverse]  # half the edges have id < reverse_id
    dv = mesh.edge_vect.take(half_edges, 1)
    short_edges = set(half_edges[np.sum(dv*dv, 0) < eps*eps])
    ids_t1=half_edges[np.sum(dv*dv, 0) < eps*eps]
    
    if not short_edges:
        return mesh, ids_t1, T1_counter, T2_counter
    reverse, vertices, face_id_by_edge = edges.reverse.copy(), mesh.vertices.copy(), mesh.face_id_by_edge.copy()
    rotate = edges.rotate
    # Do T1 transitions
    # to avoid nasty edge cases, we don't allow T1's to happen on adjacent edges
    # and delay to the next timestep if necessary.
    # A better approach would be to take multiple partial timesteps.
    boundary_edges = mesh.boundary_edges if mesh.has_boundary() else []
    #print(short_edges)
    while short_edges:
        edge = short_edges.pop()
        if edge in boundary_edges:
            edge = reverse[edge]
        neighbours = _T1(edge, eps, rotate, reverse, vertices, face_id_by_edge)
        T1_counter = T1_counter + 1
        #print("T1_counter", T1_counter)
        #time.sleep(.5)
        #print("short_edge, T1_counter, Neigbours", edge, T1_counter, neighbours)
        for x in neighbours:
            #edge_stack = np.concatenate((vertices[:,x],vertices[:,x]+mesh.edge_vect[:,x]), axis=0)  #MZ: debugging
            #print(edge_stack)
            #if short_edges:
                #print("Short edge",short_edges)
                #time.sleep(.5)
            short_edges.discard(x)      

    # Remove collapsed (ie two-sided) faces.
    while True:
        nxt = rotate[reverse]
        two_sided = np.where(nxt[nxt] == edges.ids[:len(nxt)])[0]
        stop_loop = 0 #MZ; debugging; escapes infinite loop when mesh colapses
        if not len(two_sided):  #exit if no two-sided edges are present
            break
        while np.any(reverse[reverse[rotate[two_sided]]] != reverse[rotate[nxt[two_sided]]]):
            reverse[reverse[rotate[two_sided]]] = reverse[rotate[nxt[two_sided]]]
            stop_loop = stop_loop + 1   #MZ; debugging; escapes infinite loop when mesh colapses
            if stop_loop == 1000:
                print("Cannot perform T1 transitions. Mesh failed. Simulation is stopped.")
                sys.exit(-1)        #MZ; !!!EXIT when mesh colapses
        #prev_face_id_by_edge = face_id_by_edge
        reverse, vertices, face_id_by_edge = _remove(two_sided, reverse, vertices, face_id_by_edge)
        #ids_removed = np.setdiff1d(prev_face_id_by_edge,face_id_by_edge)
        #T2_counter = T2_counter + 1   #underestimates when more than 1 two_sided are in a single time step (very rare)
        T2_counter = T2_counter + len(two_sided)/2
        #print("T2_counter", T2_counter)
        #if ~(ids_t1==np.delete(ids_t1,ids_removed)):
        #print 'Ids T1 to remove:', ids_t1, ids_removed, np.delete(ids_t1,ids_removed)
        
    mesh = mesh.copy()
    mesh.edges = Edges(reverse)
    mesh.vertices = vertices
    mesh.face_id_by_edge = face_id_by_edge
    return mesh, ids_t1, T1_counter, T2_counter


def _add_edges(mesh, edge_pairs):
    n_edge_old = len(mesh.edges)
    n_edge_new = n_edge_old + 6*len(edge_pairs)

    n_face_old = mesh.n_face        #MZ: number of cell ids before division + 1

    reverse = np.resize(mesh.edges.reverse, n_edge_new)
    vertices = np.empty((2, n_edge_new))
    vertices[:, :n_edge_old] = mesh.vertices
    face_id_by_edge = np.resize(mesh.face_id_by_edge, n_edge_new)
    rotate = Edges.ROTATE

    v = vertices.T  # easier to work with transpose here
    n = n_edge_old
    for i, (e1, e2) in enumerate(edge_pairs):
        # find midpoints: rotate[reverse] = next
        v1, v2 = 0.5*(v.take(rotate[reverse[[e1, e2]]], 0) + v.take([e1, e2], 0))
        v[[n, n+1, n+2]] = v1
        v[[n+3, n+4, n+5]] = v2

        a = [n, n+1, n+2, n+3, n+5]
        b = [e1, n+4, reverse[e1], e2, reverse[e2]]
        reverse[a], reverse[b] = b, a

        for j, edge in enumerate((e1, e2)):
            face_id = n_face_old + 2*i + j
            #print("New ids: %d" % face_id)        #MZ: debugging
            while face_id_by_edge[edge] != face_id:
                face_id_by_edge[edge] = face_id
                edge = rotate[reverse[edge]]  # rotate[reverse] = next

        re1, re2 = rotate[[e1, e2]]
        # winding
        face_id_by_edge[n] = face_id_by_edge[re1]
        v[n] += v[re1]-v[e1]
        face_id_by_edge[n+3] = face_id_by_edge[re2]
        v[n+3] += v[re2]-v[e2]
        n += 6

    mesh = mesh.copy()
    mesh.edges = Edges(reverse)
    mesh.face_id_by_edge = face_id_by_edge
    mesh.vertices = vertices
    mesh.n_face = n_face_old + 2*len(edge_pairs)
    return mesh

def _add_edges_extended(mesh, edge_pairs):
    n_edge_old = len(mesh.edges)
    n_edge_new = n_edge_old + 6*len(edge_pairs)
    #print("n_edge_new", n_edge_new)
    #print("edge_pairs", edge_pairs)
    n_face_old = mesh.n_face        #MZ: number of cell ids before division + 1

    reverse = np.resize(mesh.edges.reverse, n_edge_new)  #MZ: extending reverse array
    vertices = np.empty((2, n_edge_new))
    vertices[:, :n_edge_old] = mesh.vertices
    face_id_by_edge = np.resize(mesh.face_id_by_edge, n_edge_new)  #MZ: extending face_id_by_edge array
    rotate = Edges.ROTATE

    v = vertices.T  # easier to work with transpose here
    n = n_edge_old
    
    ids_daughters = np.empty(2*len(edge_pairs))      #MZ: added to have lineage tree
    
    for i, (e1, e2) in enumerate(edge_pairs):
        # find midpoints: rotate[reverse] = next
        v1, v2 = 0.5*(v.take(rotate[reverse[[e1, e2]]], 0) + v.take([e1, e2], 0))
        v[[n, n+1, n+2]] = v1
        v[[n+3, n+4, n+5]] = v2

        a = [n, n+1, n+2, n+3, n+5]
        b = [e1, n+4, reverse[e1], e2, reverse[e2]]
        reverse[a], reverse[b] = b, a

        for j, edge in enumerate((e1, e2)):
            face_id = n_face_old + 2*i + j
            #print("New ids: %d" % face_id)          #MZ: debugging
            ids_daughters[2*i + j] = face_id         #MZ: added to have lineage tree
            while face_id_by_edge[edge] != face_id:
                face_id_by_edge[edge] = face_id
                edge = rotate[reverse[edge]]  # rotate[reverse] = next

        re1, re2 = rotate[[e1, e2]]
        # winding
        face_id_by_edge[n] = face_id_by_edge[re1]
        v[n] += v[re1]-v[e1]
        face_id_by_edge[n+3] = face_id_by_edge[re2]
        v[n+3] += v[re2]-v[e2]
        n += 6

    mesh = mesh.copy()
    mesh.edges = Edges(reverse)
    mesh.face_id_by_edge = face_id_by_edge
    mesh.vertices = vertices
    mesh.n_face = n_face_old + 2*len(edge_pairs)
    return mesh, ids_daughters

# <codecell>


# <codecell>


