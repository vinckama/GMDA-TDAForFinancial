#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vincent Roy & Arthur Claude
"""

import pandas as pd
import numpy as np
import re
from os import path, mkdir, listdir
from shutil import rmtree
from scipy import signal
import sys
import pymannkendall as mk

class Persistence:
    """Base class for persistence computations

    Attributes:
        dataframe: dataframe on which the persistence will be calculated
    """

    DATA_PATH = 'GMDA_data/...'

    def __init__(self, dataframe, DATA_PATH='GMDA_data/.None'):
        self.df = dataframe
        self.DATA_PATH = DATA_PATH
        if not path.exists(self.DATA_PATH):
            mkdir(self.DATA_PATH)

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
        if w_size<=0:
            raise IndexError(f"w_size must be positive")
        if idx - w_size + 1 <= 0:
            raise IndexError(f"w_size > idx ({w_size} > {idx})")

    def load_dataset(self, file_name):
        file_full_name = f'.{file_name}.csv'
        if file_full_name in listdir(self.DATA_PATH):
            df = pd.read_csv(path.join(self.DATA_PATH, file_full_name))
            return df
        return None

    def save_dataset(self, df, file_name):
        df.to_csv(path.join(self.DATA_PATH, f'.{file_name}.csv'))

    def clean_dataset(self):
        if path.exists(self.DATA_PATH):
            rmtree(self.DATA_PATH)
            mkdir(self.DATA_PATH)

    def variance(self, norm):
        V = np.zeros(self.df.shape[0])
        for idx in range(500, self.df.shape[0]):
            L_window = norm[idx - 500: idx]
            var = np.var(L_window)
            V[idx] = var
        return V

    def av_spectral_density(self, norm):
        SD = np.zeros(self.df.shape[0])
        for idx in range(500, self.df.shape[0]):
            L_window = norm[idx - 500: idx]
            f, Pxx_den = signal.periodogram(L_window)
            f, Pxx_den = np.delete(f, 0), np.delete(Pxx_den, 0)
            f, Pxx_den = f[0:len(f) // 8], Pxx_den[0:len(f) // 8]
            SD[idx] = np.mean(Pxx_den)
        return SD

    def acf_firstlag(self, norm):
        AC = np.zeros(self.df.shape[0])
        for idx in range(500, self.df.shape[0]):
            L_window = norm[idx - 500: idx]
            acf = np.correlate(L_window, L_window, mode='full')
            acf = acf[acf.size // 2:]
            AC[idx] = acf[1]
        return AC

    def compute_stats(self, norm):
        V = self.variance(norm)
        SD = self.av_spectral_density(norm)
        AC = self.acf_firstlag(norm)
        return V, SD, AC

    @staticmethod
    def test_crash(norm_stats, crash_date, norm_name):
        (V, SD, AC) = norm_stats
        sys.stdout.write(f"Results of the Mann Kendall Test "
                         f"for the {norm_name}-norm (crash: {crash_date}): \n")
        MKV = mk.original_test(V)
        sys.stdout.write(
            f"Variance:          trend = {MKV.trend} |  tau = {MKV.Tau:0.4f}\n")
        MKSD = mk.original_test(SD)
        sys.stdout.write(
            f"Spectral Density:  trend = {MKSD.trend} | tau = {MKSD.Tau:0.4f}\n")
        MKAC = mk.original_test(AC)
        sys.stdout.write(
            f"Autocorrelation:   trend = {MKAC.trend} | tau = {MKAC.Tau:0.4f}\n\n")
