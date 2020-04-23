#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vincent Roy & Arthur Claude
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import gudhi as gd
import sys
from .persistence import Persistence
from utils import ProgressBar
from scipy import signal
import pymannkendall as mk

class Norm(Persistence):
    """Class to compute the norm of the persistence

    """
    sns.set_style("darkgrid")
    sns.set_palette("husl")

    def __init__(self, dataframe):
        super().__init__(dataframe, DATA_PATH = 'GMDA_data/.norm')
        self.fig = None
        self.ax1 = None

    def get_norms(self, w_size) -> tuple:
        """Compute the series of L1 and L2 norms of persistence landscapes

        Parameters:
            w_size: size of the windows for the landscapes computations

        Returns:
            L1: Norm 1
            L2: Norm 2
        """
        last_df = self.load_dataset(f'w{w_size}_norm')
        if last_df is not None:
            L1 = last_df['L1'].values.reshape(-1)
            L2 = last_df['L2'].values.reshape(-1)
            return L1, L2

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

        df = pd.DataFrame({'L1': L1, 'L2': L2}, columns = ['L1', 'L2'])
        self.save_dataset(df, f'w{w_size}_norm')
        return L1, L2

    @staticmethod
    def normalize(array) -> np.array:
        return (array-np.min(array)) / (np.max(array)-np.min(array))

    def visualise(self, w_size, start_date=None, end_date=None, save=''):
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
        L1_r['L1'] = self.normalize(L1_r['L1'])
        L2_r['L2'] = self.normalize(L2_r['L2'])
        self.ax1.plot(L1_r.index, L1_r, label = 'L1')
        self.ax1.plot(L2_r.index, L2_r, label = 'L2',
                      color = sns.husl_palette()[2])
        self.ax1.set_xlim([L2_r.index[0], L2_r.index[-1]])
        self.ax1.legend()
        sys.stdout.write(f'Plot norm of persistence landscape\n')
        sys.stdout.flush()
        plt.draw()
        if save:
            self.fig.savefig(save)
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
        dates = pd.to_datetime(self.df.index[idx_start:idx_end])
        L1_r = L1[idx_start:idx_end]
        L1_r = pd.DataFrame(L1_r, index = dates, columns = ['L1'])
        L2_r = L2[idx_start:idx_end]
        L2_r = pd.DataFrame(L2_r, index = dates, columns = ['L2'])
        return L1_r, L2_r

    def variance(self, norm):
        V = np.zeros(self.df.shape[0])
        for idx in range(500, self.df.shape[0]):
            L_window = norm[idx-500: idx]
            var = np.var(L_window)
            V[idx] = var
        return V

    def av_spectral_density(self, norm):
        SD = np.zeros(self.df.shape[0])
        for idx in range(500, self.df.shape[0]):
            L_window = norm[idx-500: idx]
            f, Pxx_den = signal.periodogram(L_window)
            f, Pxx_den = np.delete(f, 0), np.delete(Pxx_den, 0)
            f, Pxx_den = f[0:len(f) // 8], Pxx_den[0:len(f) // 8]
            SD[idx] = np.mean(Pxx_den)
        return SD

    def acf_firstlag(self, norm):
        AC = np.zeros(self.df.shape[0])
        for idx in range(500, self.df.shape[0]):
            L_window = norm[idx-500:idx]
            acf = np.correlate(L_window, L_window, mode='full')
            acf = acf[acf.size // 2:]
            AC[idx] = acf[1]
        return AC

    def visualise_crash(self, L1_stats, L2_stats, crash_date, save=''):
        (V1, SD1, AC1) = L1_stats
        (V2, SD2, AC2) = L2_stats
        idx_crash = self.df.index.get_loc(crash_date)
        x_date = pd.to_datetime(self.df.index[idx_crash - 250: idx_crash])

        fig = plt.figure(figsize=(12, 6))
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)

        ax = fig.add_subplot(2, 3, 1)
        ax.plot(x_date, V1[idx_crash - 250: idx_crash])
        ax.set_title('Variance L1')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax = fig.add_subplot(2, 3, 2)
        ax.plot(x_date, SD1[idx_crash - 250: idx_crash])
        ax.set_title('Spectrum L1')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax = fig.add_subplot(2, 3, 3)
        ax.plot(x_date, AC1[idx_crash - 250: idx_crash])
        ax.set_title('ACF(1) L1')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        color = sns.husl_palette()[2]

        ax = fig.add_subplot(2, 3, 4)
        ax.plot(x_date, V2[idx_crash - 250: idx_crash], color = color)
        ax.set_title('Variance L2')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax = fig.add_subplot(2, 3, 5)
        ax.plot(x_date, SD2[idx_crash - 250: idx_crash], color = color)
        ax.set_title('Spectrum L2')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        ax = fig.add_subplot(2, 3, 6)
        ax.plot(x_date, AC2[idx_crash - 250: idx_crash], color = color)
        ax.set_title('ACF(1) L2')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        sys.stdout.write(f'Plot norm of persistence landscape\n')
        sys.stdout.flush()
        plt.draw()
        if save:
            self.fig.savefig(save)
        plt.pause(0.001)
        input("Press [enter] to continue.")

    def test_crash(self, L1_stats, L2_stats, crash_date):
        (V1, SD1, AC1) = L1_stats
        (V2, SD2, AC2) = L2_stats

        sys.stdout.write(f"Results of the Mann Kendall Test "
                         f"for the L1-norm (crash: {crash_date}): \n")
        MKV1 = mk.original_test(V1)
        sys.stdout.write(
            f"Variance:          trend = {MKV1.trend} |  tau = {MKV1.Tau:0.4f}\n")
        MKSD1 = mk.original_test(SD1)
        sys.stdout.write(
            f"Spectral Density:  trend = {MKSD1.trend} | tau = {MKSD1.Tau:0.4f}\n")
        MKAC1 = mk.original_test(AC1)
        sys.stdout.write(
            f"Autocorrelation:   trend = {MKAC1.trend} | tau = {MKAC1.Tau:0.4f}\n\n")

        sys.stdout.write(f"Results of the Mann Kendall Test "
                         f"for the L2-norm (crash: {crash_date}): \n")
        MKV2 = mk.original_test(V2)
        sys.stdout.write(
            f"Variance:           trend = {MKV2.trend} | tau = {MKV2.Tau:0.4f}\n")
        MKSD2 = mk.original_test(SD2)
        sys.stdout.write(
            f"Spectral Density:   trend = {MKSD2.trend} | tau = {MKSD2.Tau:0.4f}\n")
        MKAC2 = mk.original_test(AC2)
        sys.stdout.write(
            f"Autocorrelation:    trend = {MKAC2.trend} | tau = {MKAC2.Tau:0.4f}\n\n")

    def compute_stats(self, norm):
        V = self.variance(norm)
        SD = self.av_spectral_density(norm)
        AC = self.acf_firstlag(norm)
        return V, SD, AC

    def crash_stats(self, w_size, crash_year='2000', test=False, plot=False,
                    save=''):
        L1, L2 = self.get_norms(w_size)
        L1_stats = self.compute_stats(L1)
        L2_stats = self.compute_stats(L2)

        crash_dict = {
            '2000': '2000-03-10',
            '2008': '2008-09-15'
        }
        crash_date = crash_dict[crash_year]

        if test or not plot:
            self.test_crash(L1_stats, L2_stats, crash_date)
        if plot:
            self.visualise_crash(L1_stats, L2_stats, crash_date, save)
