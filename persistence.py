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

sns.set_style("darkgrid")
sns.set_palette("PuBuGn_d")


class Persistance:
    def __init__(self, dataframe):
        self.df = dataframe

    @staticmethod
    def piecewise(x, birth, death):
        """
         Compute piecewise linear function

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

    def __call__(self, date, length):
        """
        compute and  plot persistence graphs

        Parameters:
            date: the first day used to compute persistence graphs
            length: the length of the time series

        Returns:
            None
        """
        # verify the date
        if not re.match(r'[1-2][0-9][0-9][0-9]\-[0-1][0-9]\-[0-3][0-9]', date):
            raise SyntaxError("The date doesn't have the right format, "
                              "please follow the format 'YYYY-MM-DD'")
        if date not in self.df.index:
            raise IndexError("Your date is not a trading day, it's not in the "
                             "dataframe, please choose another date")
        plt.clf()
        idx = self.df.index.get_loc(date)
        array_window = self.df.iloc[idx: idx + length, :].values

        fig = plt.figure('Persistence graphs', figsize = (12, 6))
        fig.tight_layout()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        fig.subplots_adjust(left=0.08, right=0.97, top=0.9, bottom=0.1)

        rips_complex = gd.RipsComplex(points = array_window)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension = 3)

        diagramm = simplex_tree.persistence(min_persistence = 0)
        gd.plot_persistence_diagram(diagramm, axes = ax1)
        land = self.persitence_landscape(diagramm, 1, 0, 0.08, 1000, 1)
        ax2.plot(land[0])
        ax2.set_title('Persistence landscape')
        ax2.set_xlabel(r'$\frac{d+b}{2}$')
        ax2.set_ylabel(r'$\frac{d-b}{2}$')
        ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
        plt.show()
