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
        L1_r['L1'] = self.normalize(L1_r['L1'])
        L2_r['L2'] = self.normalize(L2_r['L2'])
        self.ax1.plot(L1_r.index, L1_r, label = 'L1')
        self.ax1.plot(L2_r.index, L2_r, label = 'L2')
        self.ax1.set_xlim([L2_r.index[0], L2_r.index[-1]])
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
        dates = pd.to_datetime(self.df.index[idx_start:idx_end])
        L1_r = L1[idx_start:idx_end]
        L1_r = pd.DataFrame(L1_r, index = dates, columns = ['L1'])
        L2_r = L2[idx_start:idx_end]
        L2_r = pd.DataFrame(L2_r, index = dates, columns = ['L2'])
        return L1_r, L2_r
