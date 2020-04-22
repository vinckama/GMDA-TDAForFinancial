#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vincent Roy & Arthur Claude
"""
from abc import ABC

import numpy as np
import pandas as pd
from os import path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import sys
data_path = 'data'


def create_dataset():
        DowJones_pd = pd.read_csv(path.join(data_path, "DowJones.csv"),
                                  header=0, delimiter=",")
        Nasdaq_pd = pd.read_csv(
            path.join(data_path, "Nasdaq.csv"),
                                  header=0, delimiter=",")
        Russell2000_pd = pd.read_csv(
            path.join(data_path, "Russell2000.csv"),
            header=0, delimiter=",")
        SP500_pd = pd.read_csv(
            path.join(data_path, "SP500.csv"),
            header=0, delimiter=",")

        df_dict = {
            "DowJones": DowJones_pd['Adj Close'],
            "Nasdaq": Nasdaq_pd['Adj Close'],
            "Russell2000": Russell2000_pd['Adj Close'],
            "SP500": SP500_pd['Adj Close']
        }

        for (name, df) in df_dict.items():
            df_dict[name] = np.asarray(df)

        data_df = DateDataFrame(df_dict, index = DowJones_pd.Date)
        data_df.sort_index(inplace = True)
        return data_df


class DateDataFrame(pd.DataFrame, ABC):
    """
    pd.DataFrame extended class with visualise functions
    """
    def __init__(self, *args, **kwargs):
        # use the __init__ method from DataFrame to ensure
        # that we're inheriting the correct behavior
        super(DateDataFrame, self).__init__(*args, **kwargs)
        self.fig = None

    # this method is makes it so our methods return an instance
    # of DateDataFrame, instead of a regular DataFrame
    @property
    def _constructor(self):
        return DateDataFrame

    def visualise(self, title='Stock market developments'):
        """
        visualise without subplot the dataframe
        :param title: title of the graph
        :return:
        """
        if self.fig is None:
            sns.set_style("darkgrid")
            sns.set_palette("husl")

            self.fig = plt.figure("stock market", figsize = (12, 6))
            self.fig.tight_layout()
            self.fig.subplots_adjust(left = 0.08, right = 0.97,
                                     top = 0.9, bottom = 0.1)
        else:
            plt.clf()

        self.fig.suptitle(title)
        ax1 = self.fig.add_subplot(1, 1, 1)
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        ax1.xaxis.set_major_locator(locator)
        ax1.xaxis.set_major_formatter(formatter)
        dates = pd.to_datetime(self.index)
        for column in self.columns:
            ax1.plot(dates, self.loc[:, column], label = column)
        ax1.set_xlim([dates[0], dates[-1]])
        ax1.legend()
        plt.show()
        sys.stdout.write(f'Plot {title}\n')
        sys.stdout.flush()

    def visualise_subplots(self,
            title='Stock market(?? log consecutive developpment ratio ??)'):
        """
        visualise with subplots the dataframe
        :param title: title of the graphs
        :return:
        """
        if self.fig is None:
            sns.set_style("darkgrid")
            sns.set_palette("husl")

            self.fig = plt.figure("stock market subplot", figsize = (12, 6))
            self.fig.tight_layout()
            self.fig.subplots_adjust(left = 0.08, right = 0.97,
                                     top = 0.9, bottom = 0.1)
        else:
            plt.clf()

        self.fig.suptitle(title)
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        formatter = mdates.ConciseDateFormatter(locator)
        dates = pd.to_datetime(self.index)
        graph_n = len(self.columns)
        for idx, column in enumerate(self.columns):
            ax = self.fig.add_subplot(graph_n, 1, idx + 1)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)
            ax.set_title(column, pad=-12)
            ax.plot(dates, self.loc[:, column], color=sns.husl_palette()[idx])
            ax.set_xlim([dates[0], dates[-1]])
            ax.set_ylim([self.values.min(), self.values.max()])
            plt.show()
        sys.stdout.write(f'Plot {title}\n')
        sys.stdout.flush()


