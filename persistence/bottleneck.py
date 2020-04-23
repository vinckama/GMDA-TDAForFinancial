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


class Bottleneck(Persistence):
    """Class to compute the bottleneck distance

    """
    sns.set_style("darkgrid")
    sns.set_palette("husl")

    def __init__(self, dataframe):
        super().__init__(dataframe, DATA_PATH = 'GMDA_data/.bottleneck')
        self.fig = None
        self. ax1 = None

    def get_bottleneck_distance(self, w_size):
        """Compute bottleneck distance of persistence landscape

        Parameters:
            w_size: size of the windows for the landscapes computations

        Returns:
            bottleneck: bottleneck distance
        """
        last_df = self.load_dataset(f'w{w_size}_bottleneck')
        if last_df is not None:
            bottleneck = last_df['bottleneck'].values.reshape(-1)
            return bottleneck

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

        df = pd.DataFrame({'bottleneck': bottleneck}, columns = ['bottleneck'])
        self.save_dataset(df, f'w{w_size}_bottleneck')
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
        self.ax1.plot(bottleneck_r.index, bottleneck_r)
        self.ax1.set_xlim([bottleneck_r.index[0], bottleneck_r.index[-1]])
        sys.stdout.write(f'Plot norm of bottleneck distance\n')
        sys.stdout.flush()
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")

    def __call__(self, w_size, start_date=None, end_date=None) -> pd.DataFrame:
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
        dates = pd.to_datetime(self.df.index[idx_start:idx_end])
        bottleneck_r = pd.DataFrame(bottleneck_r, index = dates,
                                    columns = ['bottleneck'])
        return bottleneck_r
