# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

"""
The :mod:`forces` module implements a selection of force laws
for vertex models.
"""

# ====Forces====

from operator import add
import numpy as np
import functools #MZ: python2 -> python3


# ====Base classes====

class Force(object):
    """
    Base interface for the different force laws.
    """
    def energy(self, cells):
        raise Exception('energy undefined')

    def force(self, cells):
        raise Exception('force undefined')
    
    #def force_noise(self, cells):
    #    raise Exception('force undefined')

    def __call__(self, cells):
        return self.force(cells)

    def __add__(self, other):
        return Combined([self, other])

    def __repr__(self):
        return type(self).__name__


class Combined(Force):
    """
    Sum of forces.
    """
    def __init__(self, summands):
        expanded = (x.summands if hasattr(x, 'summands') else [x] for x in summands)
        self.summands = sum(expanded, [])

    def energy(self, cells):
        return functools.reduce(add, (s.energy(cells) for s in self.summands))

    def force(self, cells):
        return functools.reduce(add, (s.force(cells) for s in self.summands))

    def __repr__(self):
        return ' + '.join(map(repr, self.summands))


# ====Force definitions====

class TargetArea(Force):
    r"""
    .. math:: E = \sum_{\alpha \in cells} \frac{K_\alpha}{2} (A_\alpha-A^{(0)}_\alpha)^2

    where :math:`A_\alpha` is the area of cell :math:`\alpha`.
    """
    def energy(self, cells):
        return 0.5*np.sum(cells.by_face('K')*(cells.mesh.area-cells.by_face('A0'))**2)

    def force(self, cells):
        return ((cells.by_face('K')*(cells.by_face('A0')-cells.mesh.area))
                [cells.mesh.face_id_by_edge]*cells.mesh.d_area)


class Tension(Force):
    r"""
    .. math:: E = \sum_{<ij> \in edges} \Lambda_{ij} l_{ij}

    where :math:`l_{ij}` is the length of the (undirected) edge :math:`<ij>`.
    """
    def energy(self, cells):
        #return 0.5*np.sum(cells.by_edge('Lambda', 'Lambda_boundary')*cells.mesh.length)
        return 0.5*np.sum(cells.by_edge_removal('Lambda', 'Lambda_boundary')*cells.mesh.length)
        # 0.5 since we sum over directed edges

    def force(self, cells):
        #F = (0.5*cells.by_edge('Lambda', 'Lambda_boundary')/cells.mesh.length)*cells.mesh.edge_vect
        #F = (0.5*cells.by_edge_removal('Lambda', 'Lambda_boundary')/cells.mesh.length)*cells.mesh.edge_vect
        F = (0.5*cells.by_edge_removal_Lambda_noise('Lambda_edge')/cells.mesh.length)*cells.mesh.edge_vect
        return F - F.take(cells.mesh.edges.prev, 1)


class Perimeter(Force):
    r"""
    .. math:: E = \sum_{\alpha \in cells} \frac{\Gamma_\alpha}{2} L_\alpha^2

    where :math:`L_\alpha` is the perimeter of cell :math:`\alpha`.
    """
    def energy(self, cells):
        return 0.5*np.sum(cells.by_face('Gamma')*cells.mesh.perimeter**2)

    def force(self, cells):
        return (cells.by_face('Gamma')*cells.mesh.perimeter)[cells.mesh.face_id_by_edge]*cells.mesh.d_perimeter


class Pressure(Force):
    r"""
    .. math:: E = -\sum_{\alpha\in cells} P_\alpha A_\alpha

    where :math:`A_\alpha` is the area of cell :math:`\alpha`.
    """
    def energy(self, cells):
    #warning - can't compute an energy with non-zero external pressure (would be infinite)
        return -np.sum(cells.by_face('P')*cells.mesh.area)

    def force(self, cells):
        return cells.by_edge('P','boundary_P')*cells.mesh.d_area


class Hooke(Force):
    r"""
    .. math:: E = \sum_{<ij> \in edges} T_{ij} (L_{ij}^{(0)} - l_{ij})^2

    where :math:`l_{ij}` is the length of the (undirected) edge :math:`<ij>`.
    """

    def energy(self, cells):
        return 0.5*np.sum(cells.by_edge('T')*(cells.mesh.length-cells.by_edge('L0'))**2)

    def force(self,  cells):
        F = cells.by_edge('T')*(1.0-cells.by_edge('L0')/cells.mesh.length)*cells.mesh.edge_vect
        return F - F.take(cells.mesh.edges.prev, 1)

