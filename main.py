#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vincent Roy & Arthur Claude
"""

from data import load_data
import matplotlib.pyplot as plt
from utils import log_df
from persistence import Landscape, Norm


def visualise(df):
    df.plot()
    plt.xticks(rotation = 20)




if __name__ == "__main__":
    df = load_data()
    # visualise(df)
    df_log = log_df(df)
    # visualise(df_log)
    aa = Landscape(df_log)
    #aa('2000-03-10', 80)

    nn = Norm(df_log)
    #nn('1997-01-03', '2000-05-10', 50)
