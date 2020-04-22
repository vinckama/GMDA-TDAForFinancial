#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vincent Roy & Arthur Claude
"""

from data import load_data
import matplotlib.pyplot as plt
import math
from persistence import Landscape, Norm


def visualise(df):
    df.plot()
    plt.xticks(rotation = 20)


def visualise_subplot(df):
    sub_n = len(df.columns)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax2 = fig.add_subplot()
    for k in range(10):
        # axes.subplot(sub_n, 1, k)
        pass


def log_df(df):
    shifted_df = df.shift(-1)
    ratio_df = df / shifted_df
    ratio_df.dropna(inplace = True)
    return ratio_df.applymap(lambda x: math.log(x))


if __name__ == "__main__":
    df = load_data()
    # visualise(df)
    df_log = log_df(df)
    # visualise(df_log)
    aa = Landscape(df_log)
    #aa('2000-03-10', 80)

    nn = Norm(df_log)
    #nn('1997-01-03', '2000-05-10', 50)
