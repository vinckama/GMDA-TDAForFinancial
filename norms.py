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


class Norms:
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

    def get_norms(self, w):
        """
        Compute the series of L1 and L2 norms of persistence landscapes

        Parameters:
            w: size of the windows for the landscapes computations

        Returns:
            L1:
            L2:
        """
        length = self.df.shape[0]
        L1,L2 = np.zeros(length),np.zeros(length)
        for idx in range(w, length):
            array_window = self.df.iloc[idx-w: idx, :].values
            rips_complex = gd.RipsComplex(points = array_window)
            simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
            diagram = simplex_tree.persistence(min_persistence=0)
            land = self.persitence_landscape(diagram,1,0,0.08,1000,1)
            norm1 = np.linalg.norm(land, ord=1)
            norm2 = np.linalg.norm(land)
            L1[idx] = norm1
            L2[idx] = norm2
        return L1,L2

    def verify_date(self, date):
        if not re.match(r'[1-2][0-9][0-9][0-9]\-[0-1][0-9]\-[0-3][0-9]', date):
            raise SyntaxError("The date doesn't have the right format, "
                              "please follow the format 'YYYY-MM-DD'")
        if date not in self.df.index:
            raise IndexError("Your date is not a trading day, it's not in the "
                             "dataframe, please choose another date")

    @staticmethod
    def normalize(array) -> np.array:
        return (array-np.min(array)) / (np.max(array)-np.min(array))

    def __call__(self, start_date, end_date, length):
        '''
        Plot L1 and L2 norms series

        Parameters:
            start_date: start date of the period studied
            end_date: end date of the period studied
            w: size of the windows for the landscapes computation
        '''
        self.verify_date(start_date)
        self.verify_date(end_date)
        plt.clf()

        fig = plt.figure(' Norm of persistence landscape', figsize = (12, 6))
        fig.tight_layout()
        ax1 = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0.08, right=0.97, top=0.9, bottom=0.1)

        L1,L2 = self.get_norms(length)
        idx_start = self.df.index.get_loc(start_date)
        idx_end = self.df.index.get_loc(end_date)
        L1_r, L2_r = L1[idx_start:idx_end],L2[idx_start:idx_end]
        L1_r_normalized = self.normalize(L1_r)
        L2_r_normalized = self.normalize(L2_r)

        ax1.plot(L1_r_normalized, label = 'L1')
        ax1.plot(L2_r_normalized, label = 'L2')
        ax1.set_title('Normalized L1 and L2 norms of persistence landscapes')
        ax1.legend()
        plt.show()


