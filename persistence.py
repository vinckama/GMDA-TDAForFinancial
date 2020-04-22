#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vincent Roy & Arthur Claude
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pandas as pd
import numpy as np
import gudhi as gd
import re


class Persistence:
    """Base class for persistence computations

    Attributes:
        dataframe: dataframe on which the persistence will be calculated
    """
    def __init__(self, dataframe):
        self.df = dataframe

    @staticmethod
    def piecewise(x, birth, death):
        """Compute piecewise linear function

        Parameters:
            x: point
            birth: birth of the point
            death: death of the point

        Returns:
            a float
        """
        if birth < x <= (birth + death)/2:
            return x - birth
        elif (birth + death)/2 < x < death:
            return - x + death
        else:
            return 0

    def persitence_landscape(self, dgm, k, x_min, x_max, nb_nodes, nb_ld):
        """
        Compute persistence landscapes

        Parameters:
            dgm: a persistence diagram in the Gudhi format
            k: dimension
            x_min: first endpoint of an interval
            x_max: second endpoint of an interval
            nb_nodes: number of nodes of a regular grid on [x_min,x_max]
            nb_ld: number of landscapes

        Returns:
            a nb_ld*nb_nodes array storing the values of the first
            nb_ld landscapes of dgm on the nodes of the grid
        """
        x_seq = np.linspace(x_min, x_max, nb_nodes)

        flatdgm = []
        for i in dgm:
            flatdgmi = [i[0], i[1][0], i[1][1]]
            flatdgm.append(flatdgmi)
        df_dgm = pd.DataFrame(flatdgm, columns = ['dim', 'birth', 'death'])
        df_dgm_dim = df_dgm.loc[df_dgm['dim'] == k]
        nb_rows = df_dgm_dim.shape[0]

        if nb_rows == 0:
            return np.repeat(0, nb_nodes)

        f = np.zeros((nb_nodes, nb_rows))

        for i in range(nb_rows):
            birth = df_dgm.iloc[i]['birth']
            death = df_dgm.iloc[i]['death']

            for j in range(nb_nodes):
                f[j][i] = self.piecewise(x_seq[j], birth, death)
        for j in range(nb_nodes):
            f[j] = sorted(f[j], reverse=True)

        landscapes = np.nan_to_num(np.transpose(f[:, :nb_ld]))

        return landscapes

    def verify_date(self, date):
        if not re.match(r'[1-2][0-9][0-9][0-9]\-[0-1][0-9]\-[0-3][0-9]', date):
            raise SyntaxError("The date doesn't have the right format, "
                              "please follow the format 'YYYY-MM-DD'")
        if date not in self.df.index:
            raise IndexError("Your date is not a trading day, it's not in the "
                             "dataframe, please choose another date")

    @staticmethod
    def verify_w_size(idx, w_size):
        if idx - w_size + 1 <= 0:
            raise IndexError(f"w_size > idx ({w_size} > {idx})")


class Norm(Persistence):
    """Class to compute the norm of the persistence
    
    """
    sns.set_style("darkgrid")
    sns.set_palette("husl")

    def __init__(self, dataframe):
        super().__init__(dataframe)
        self.fig = None
        self. ax1 = None
        self.last_w_size = None
        self.last_L1 = None
        self.last_L2 = None

    def get_norms(self, w_size) -> tuple:
        """Compute the series of L1 and L2 norms of persistence landscapes

        Parameters:
            w_size: size of the windows for the landscapes computations

        Returns:
            L1: Norm 1
            L2: Norm 2
        """

        if w_size == self.last_w_size:
            return self.last_L1, self.last_L2

        else:
            length = self.df.shape[0]
            L1, L2 = np.zeros(length), np.zeros(length)
            for idx in range(w_size, length):
                array_window = self.df.iloc[idx - w_size: idx, :].values
                rips_complex = gd.RipsComplex(points = array_window)
                simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
                diagram = simplex_tree.persistence(min_persistence=0)
                land = self.persitence_landscape(diagram, 1, 0, 0.08, 1000, 1)
                norm1 = np.linalg.norm(land, ord=1)
                norm2 = np.linalg.norm(land)
                L1[idx] = norm1
                L2[idx] = norm2

                if idx % 100 ==0:
                    print(f"{idx}/{length}")
            return L1, L2

    @staticmethod
    def normalize(array) -> np.array:
        return (array-np.min(array)) / (np.max(array)-np.min(array))

    def plot_norm(self, start_date, end_date, w_size):
        """Plot L1 and L2 norms series

        Parameters:
            start_date: start date of the period studied
            end_date: end date of the period studied
            w_size: size of the windows for the landscapes computation
        """
        if self.fig is None:
            self.fig = plt.figure(' Norm of persistence landscape',
                                  figsize = (12, 6))
            self.fig.tight_layout()
            self.fig.subplots_adjust(left=0.08, right=0.97, top=0.9, bottom=0.1)
            self.ax1 = self.fig.add_subplot(1, 1, 1)
            self.ax1.set_title('Normalized L1 and L2 norms of '
                               'persistence landscapes')
        else:
            self.ax1.lines = []

        L1_r, L2_r = self.__call__(start_date, end_date, w_size)
        L1_r_normalized = self.normalize(L1_r)
        L2_r_normalized = self.normalize(L2_r)

        self.ax1.plot(L1_r_normalized, label = 'L1')
        self.ax1.plot(L2_r_normalized, label = 'L2')
        self.ax1.legend()
        self.fig.show()

    def __call__(self, start_date, end_date, w_size) -> tuple:
        """Compute L1 and L2 norms series

        Parameters:
            start_date: start date of the period studied
            end_date: end date of the period studied
            w_size: size of the windows for the landscapes computation
        """
        self.verify_date(start_date)
        self.verify_date(end_date)

        L1, L2 = self.get_norms(w_size)
        idx_start = self.df.index.get_loc(start_date)
        idx_end = self.df.index.get_loc(end_date)
        L1_r = L1[idx_start:idx_end]
        L2_r = L2[idx_start:idx_end]
        return L1_r, L2_r


class Landscape(Persistence):
    """
    class to compute the landscape of the persistence
    """
    sns.set_style("darkgrid")
    sns.set_palette("husl")

    def __init__(self, dataframe):
        super().__init__(dataframe)
        self.fig = None
        self.ax1 = None
        self.ax2 = None

    def plot_landscape(self, end_date, w_size) -> None:
        """"
        plot persistence and its landscape

        Parameters:
            end_date: the last day used to compute persistence graphs
            w_size: size of the windows for the landscapes computations

        Returns:
            None
        """
        if self.fig is None:
            self.fig = plt.figure('Persistence graphs', figsize = (12, 6))
            self.fig.tight_layout()
            self.fig.subplots_adjust(left=0.08, right=0.97, top=0.9, bottom=0.1)
            self.ax1 = self.fig.add_subplot(1, 2, 1)

            self.ax2 = self.fig.add_subplot(1, 2, 2)
            self.ax2.set_title('Persistence landscape')
            self.ax2.set_xlabel(r'$\frac{d+b}{2}$')
            self.ax2.set_ylabel(r'$\frac{d-b}{2}$')
            self.ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        else:
            self.ax1.cla()
            self.ax2.lines = []

        diagram, land = self.__call__(end_date, w_size)
        gd.plot_persistence_diagram(diagram, axes = self.ax1)
        self.ax2.plot(land[0])
        self.fig.show()

    @staticmethod
    def __min_birth_max_death(persistence, band=0.0):
        """ This function returns (min_birth, max_death) from the persistence.

        function from the Gudhi module (https://gudhi.inria.fr)
        :param persistence: The persistence to plot.
        :type persistence: list of tuples(dimension, tuple(birth, death)).
        :param band: band
        :type band: float.
        :returns: (float, float) -- (min_birth, max_death).
        """
        # Look for minimum birth date and maximum death date for plot optimisation
        max_death = 0
        min_birth = persistence[0][1][0]
        for interval in reversed(persistence):
            if float(interval[1][1]) != float("inf"):
                if float(interval[1][1]) > max_death:
                    max_death = float(interval[1][1])
            if float(interval[1][0]) > max_death:
                max_death = float(interval[1][0])
            if float(interval[1][0]) < min_birth:
                min_birth = float(interval[1][0])
        if band > 0.0:
            max_death += band
        return min_birth, max_death

    def find_x_max(self, persistence, inf_delta=0.1, band=0.0):
        """Replace infinity values with max_death + delta for diagram to be more

        :param persistence: The persistence to plot.
        :param inf_delta: the delta of the infiny.
        :param band: band
        :returns: float -- x_max
        """

        (min_birth, max_death) = self.__min_birth_max_death(persistence, band)
        delta = (max_death - min_birth) * inf_delta
        x_max = max_death + delta
        return x_max

    def __call__(self, end_date, w_size) -> tuple:
        """
        compute persistence and its landscape

        Parameters:
            end_date: the last day used to compute persistence graphs
            w_size: size of the windows for the landscapes computations

        Returns:
            (persistence, landscape)
        """
        self.verify_date(end_date)
        idx = self.df.index.get_loc(end_date)
        self.verify_w_size(idx, w_size)

        array_window = self.df.iloc[idx - w_size + 1: idx + 1, :].values
        rips_complex = gd.RipsComplex(points = array_window)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension = 2)

        diagram = simplex_tree.persistence(min_persistence = 0)
        x_max = self.find_x_max(diagram)
        land = self.persitence_landscape(diagram, 1, 0, x_max, 1000, 1)

        return diagram, land


