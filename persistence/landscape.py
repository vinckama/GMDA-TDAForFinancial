#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vincent Roy & Arthur Claude
"""


import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import gudhi as gd
import sys
from .persistence import Persistence


class Landscape(Persistence):
    """
    class to compute the landscape of the persistence
    """
    sns.set_style("darkgrid")
    sns.set_palette("husl")

    def __init__(self, dataframe):
        super().__init__(dataframe, DATA_PATH = 'GMDA_data/.landscapes')
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
