#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vincent Roy & Arthur Claude
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
import gudhi as gd
import re
import sys
from utils import ProgressBar


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
            raise IndexError(f"Your date ({date}) is not a trading day, "
                             f"it's not in the dataframe, "
                             f"please choose another date")

    @staticmethod
    def verify_w_size(idx, w_size):
        if idx - w_size + 1 <= 0:
            raise IndexError(f"w_size > idx ({w_size} > {idx})")


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

    def visualise(self, w_size, end_date=None) -> None:
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

        diagram, land = self.__call__(w_size, end_date)
        gd.plot_persistence_diagram(diagram, axes = self.ax1)
        self.ax2.plot(land[0])
        sys.stdout.write(f'Plot Persistence graphs\n')
        sys.stdout.flush()
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")

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
        :param inf_delta: the delta of the infinity.
        :param band: band
        :returns: float -- x_max
        """

        (min_birth, max_death) = self.__min_birth_max_death(persistence, band)
        delta = (max_death - min_birth) * inf_delta
        x_max = max_death + delta
        return x_max

    def __call__(self, w_size, end_date=None) -> tuple:
        """
        compute persistence and its landscape

        Parameters:
            end_date: the last day used to compute persistence graphs
            w_size: size of the windows for the landscapes computations

        Returns:
            (persistence, landscape)
        """
        if end_date is not None:
            self.verify_date(end_date)
            idx = self.df.index.get_loc(end_date)
        else:
            idx = self.df.index.shape[0]
        self.verify_w_size(idx, w_size)

        array_window = self.df.iloc[idx - w_size + 1: idx + 1, :].values
        rips_complex = gd.RipsComplex(points = array_window)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension = 2)

        diagram = simplex_tree.persistence(min_persistence = 0)
        x_max = self.find_x_max(diagram)
        land = self.persitence_landscape(diagram, 1, 0, x_max, 1000, 1)

        return diagram, land


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

        length = self.df.shape[0] - w_size
        L1, L2 = np.zeros(self.df.shape[0]), np.zeros(self.df.shape[0])

        message = f"Compute the norm for a window of {w_size} " \
            f"on {length} points\n"
        sys.stdout.write(message + "-"* (len(message) -1) + '\n')
        sys.stdout.flush()
        pb = ProgressBar(total = length)

        for idx in range(length):
            array_window = self.df.iloc[idx: idx + w_size, :].values
            rips_complex = gd.RipsComplex(points = array_window)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
            diagram = simplex_tree.persistence(min_persistence=0)
            land = self.persitence_landscape(diagram, 1, 0, 0.08, 1000, 1)
            norm1 = np.linalg.norm(land, ord=1)
            norm2 = np.linalg.norm(land)
            L1[idx + w_size] = norm1
            L2[idx + w_size] = norm2
            next(pb)

        self.last_w_size = w_size
        self.last_L1 = L1
        self.last_L2 = L2
        return L1, L2

    @staticmethod
    def normalize(array) -> np.array:
        return (array-np.min(array)) / (np.max(array)-np.min(array))

    def visualise(self, w_size, start_date=None, end_date=None):
        """Plot L1 and L2 norms series on a time window

        Parameters:
            start_date: start date of the period studied
            end_date: end date of the period studied
            w_size: size of the windows for the landscapes computation
        """
        if self.fig is None:
            self.fig = plt.figure('Norm of persistence landscape',
                                  figsize = (12, 6))
            self.fig.tight_layout()
            self.fig.subplots_adjust(left=0.08, right=0.97, top=0.9, bottom=0.1)
            self.ax1 = self.fig.add_subplot(1, 1, 1)
            self.ax1.set_title('Normalized L1 and L2 norms of '
                               'persistence landscape')
            locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
            formatter = mdates.ConciseDateFormatter(locator)
            self.ax1.xaxis.set_major_locator(locator)
            self.ax1.xaxis.set_major_formatter(formatter)
        else:
            self.ax1.lines = []

        L1_r, L2_r = self.__call__(w_size, start_date, end_date)
        L1_r_normalized = self.normalize(L1_r)
        L2_r_normalized = self.normalize(L2_r)
        dates = pd.to_datetime(self.df.loc[start_date: end_date].index[:-1])
        self.ax1.plot(dates, L1_r_normalized, label = 'L1')
        self.ax1.plot(dates, L2_r_normalized, label = 'L2')
        self.ax1.set_xlim([dates[0], dates[-1]])
        self.ax1.legend()
        sys.stdout.write(f'Plot norm of persistence landscape\n')
        sys.stdout.flush()
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")

    def __call__(self, w_size, start_date=None, end_date=None) -> tuple:
        """Compute L1 and L2 norms series on a time window

        Parameters:
            start_date: start date of the period studied
            end_date: end date of the period studied
            w_size: size of the windows for the landscapes computation
        """
        if start_date is not None:
            self.verify_date(start_date)
            idx_start = self.df.index.get_loc(start_date)
        else:
            idx_start = w_size
        if end_date is not None:
            self.verify_date(end_date)
            idx_end = self.df.index.get_loc(end_date)
        else:
            idx_end = self.df.index.shape[0]
        self.verify_w_size(idx_start, w_size)

        L1, L2 = self.get_norms(w_size)
        L1_r = L1[idx_start:idx_end]
        L2_r = L2[idx_start:idx_end]
        return L1_r, L2_r


class Bottleneck(Persistence):
    """Class to compute the bottleneck distance

    """
    sns.set_style("darkgrid")
    sns.set_palette("husl")

    def __init__(self, dataframe):
        super().__init__(dataframe)
        self.fig = None
        self. ax1 = None
        self.last_w_size = None
        self.last_bottleneck = None

    def get_bottleneck_distance(self, w_size):
        """Compute bottleneck distance of persistence landscape

        Parameters:
            w_size: size of the windows for the landscapes computations

        Returns:
            bottleneck: bottleneck distance
        """
        if w_size == self.last_w_size:
            return self.last_bottleneck

        length = self.df.shape[0] - w_size
        bottleneck = np.zeros(self.df.shape[0])
        prev_diagram_b = None

        message = f"Compute the bottleneck distance for a window of {w_size} " \
                f"on {length} points\n"
        sys.stdout.write(message + "-"* (len(message) -1) + '\n')
        sys.stdout.flush()
        pb = ProgressBar(total = length)

        for idx in range(length):
            array_window = self.df.iloc[idx: idx + w_size, :].values
            rips_complex = gd.RipsComplex(points = array_window)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
            current_diagram = simplex_tree.persistence(min_persistence=0)

            current_diagram_b = []
            for i in range(len(current_diagram)):
                if current_diagram[i][0] == 1:
                    current_diagram_b.append([current_diagram[i][1][0],
                                              current_diagram[i][1][1]])

            if prev_diagram_b != None:
                dist = gd.bottleneck_distance(current_diagram_b, prev_diagram_b)
                bottleneck[idx + w_size] = dist
            prev_diagram_b = current_diagram_b
            next(pb)

        self.last_w_size = w_size
        self.last_bottleneck = bottleneck
        return bottleneck

    def visualise(self, w_size, start_date=None, end_date=None):
        """Plot bottleneck distance on a time window

        Parameters:
            start_date: start date of the period studied
            end_date: end date of the period studied
            w_size: size of the windows for the landscapes computation
        """
        if self.fig is None:
            self.fig = plt.figure('Bottleneck of persistence landscape',
                                  figsize = (12, 6))
            self.fig.tight_layout()
            self.fig.subplots_adjust(left=0.08, right=0.97, top=0.9, bottom=0.1)
            self.ax1 = self.fig.add_subplot(1, 1, 1)
            self.ax1.set_title('Bottleneck distance of '
                               'persistence landscape')
            locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
            formatter = mdates.ConciseDateFormatter(locator)
            self.ax1.xaxis.set_major_locator(locator)
            self.ax1.xaxis.set_major_formatter(formatter)
        else:
            self.ax1.lines = []

        bottleneck_r = self.__call__(w_size, start_date, end_date)
        dates = pd.to_datetime(self.df.loc[start_date: end_date].index[:-1])
        self.ax1.plot(dates, bottleneck_r)
        self.ax1.set_xlim([dates[0], dates[-1]])
        sys.stdout.write(f'Plot norm of bottleneck distance\n')
        sys.stdout.flush()
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")

    def __call__(self, w_size, start_date=None, end_date=None) -> tuple:
        """Compute bottleneck distance on a time window

        Parameters:
            start_date: start date of the period studied
            end_date: end date of the period studied
            w_size: size of the windows for the landscapes computation
        """
        if start_date is not None:
            self.verify_date(start_date)
            idx_start = self.df.index.get_loc(start_date)
        else:
            idx_start = w_size
        if end_date is not None:
            self.verify_date(end_date)
            idx_end = self.df.index.get_loc(end_date)
        else:
            idx_end = self.df.index.shape[0]
        self.verify_w_size(idx_start, w_size)

        bottleneck = self.get_bottleneck_distance(w_size)
        bottleneck_r = bottleneck[idx_start:idx_end]
        return bottleneck_r
