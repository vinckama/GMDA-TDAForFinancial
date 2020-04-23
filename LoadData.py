#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vincent Roy & Arthur Claude
"""
import numpy as np
from abc import ABC
import pandas as pd
from os import path, mkdir, listdir
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import sys
from math import log
import requests
import io


class DataLoader:
    DATA_PATH = 'GMDA_data'
    dataset_dict = {
    'DowJones':'https://geometrica.saclay.inria.fr/team/Fred.Chazal'
               '/Projects/TDAFinancialTimeSeries/Yahoo_finance/DowJones.csv',
    'Nasdaq':'https://geometrica.saclay.inria.fr/team/Fred.Chazal'
             '/Projects/TDAFinancialTimeSeries/Yahoo_finance/Nasdaq.csv',
    'Russell2000':'https://geometrica.saclay.inria.fr/team/Fred.Chazal'
                  '/Projects/TDAFinancialTimeSeries/Yahoo_finance'
                  '/Russell2000.csv',
    'SP500':'https://geometrica.saclay.inria.fr/team/Fred.Chazal'
            '/Projects/TDAFinancialTimeSeries/Yahoo_finance/SP500.csv',
}

    def __init__(self):
        self.df = self.create_dataset()
        self.log_df = self.create_log_df()

    @staticmethod
    def download_data(url):
        re = requests.get(url, verify=False)
        if re.ok:
            return pd.read_csv(io.StringIO(re.content.decode()))
        else:
            return None
        
    def verify_data_path(self):
        if not path.exists(self.DATA_PATH):
            mkdir(self.DATA_PATH)
            sys.stdout.write(f"Creating the folder '{self.DATA_PATH}'"
                             f" to hold the data.")
            sys.stdout.flush()
        data_path_files= listdir(self.DATA_PATH)
        for dataset in self.dataset_dict.keys():
            if f'{dataset}.csv' not in data_path_files:
                df = self.download_data(self.dataset_dict[dataset])
                if df is not None:
                    sys.stdout.write(f'{dataset} downloaded\n')
                    sys.stdout.flush()
                    df.to_csv(path.join(self.DATA_PATH, f'{dataset}.csv'))
                else:
                    raise FileNotFoundError(f"Please add '{dataset}.csv' "
                                            f"in the '{self.DATA_PATH}' folder")

    def load_dataset(self, file_name):
        file_full_name = f'.{file_name}.csv'
        if file_full_name in listdir(self.DATA_PATH):
            df = DateDataFrame(
                pd.read_csv(path.join(self.DATA_PATH, file_full_name)))
            df.set_index('Date', inplace = True)
            return df
        return None

    def save_dataset(self, df, file_name):
        df.to_csv(path.join(self.DATA_PATH, f'.{file_name}.csv'))

    def create_dataset(self):
        self.verify_data_path()
        saved_df = self.load_dataset("data_df")
        if saved_df is not None:
            return saved_df
        sys.stdout.write("Create the dataset\n")
        sys.stdout.flush()

        df_dict = dict()
        for dataset in self.dataset_dict.keys():
            df = pd.read_csv(path.join(self.DATA_PATH, f"{dataset}.csv"),
                                      header=0, delimiter=",")
            df_dict[dataset] = df['Adj Close']
            index = df['Date']

        for (name, df) in df_dict.items():
            df_dict[name] = np.asarray(df)

        data_df = DateDataFrame(df_dict, index = index)[:1000] #################################@
        data_df.sort_index(inplace = True)
        self.save_dataset(data_df, 'data_df')
        return data_df

    def create_log_df(self):
        saved_df = self.load_dataset("ratio_df")
        if saved_df is not None:
            return saved_df
        sys.stdout.write("Create the ratio-dataset\n")
        sys.stdout.flush()

        shifted_df = self.df.shift(-1)
        ratio_df = self.df / shifted_df
        ratio_df.dropna(inplace = True)
        ratio_df = ratio_df.applymap(lambda x: log(x))
        self.save_dataset(ratio_df, 'ratio_df')
        return ratio_df

    def __call__(self):
        return self.df, self.log_df


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
        sys.stdout.write(f'Plot {title}\n')
        sys.stdout.flush()
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")


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
        sys.stdout.write(f'Plot {title}\n')
        sys.stdout.flush()
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")




