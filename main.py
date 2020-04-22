#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vincent Roy & Arthur Claude
"""

from data import create_dataset
from utils import log_df
from persistence import Landscape, Norm, Bottleneck

if __name__ == "__main__":
    df = create_dataset()
    df.visualise()

    df_log = log_df(df)
    df_log.visualise_subplots()

    landscape = Landscape(df_log)
    landscape.visualise('2000-03-10', 80)

    norm = Norm(df_log[:1000])
    norm.visualise('1988-01-25', '1991-08-06', 20)

    bottleneck = Bottleneck(df_log[:1000])
    bottleneck.visualise('1988-01-25', '1991-08-06', 20)
    nn = Norm(df_log)
    #nn.plot_norm('1997-01-03', '2000-05-10', 50)
