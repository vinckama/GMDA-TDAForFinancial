#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vincent Roy @ Arthur Claude
"""
import numpy as np
import pandas as pd
from os import path


def load_data(data_path="data"):
    DowJones_pd = pd.read_csv(path.join(data_path, "DowJones.csv"),
                              header=0, delimiter=",")
    Nasdaq_pd = pd.read_csv(path.join(data_path, "Nasdaq.csv"),
                              header=0, delimiter=",")
    Russell2000_pd = pd.read_csv(path.join(data_path, "Russell2000.csv"),
                              header=0, delimiter=",")
    SP500_pd = pd.read_csv(path.join(data_path, "SP500.csv"),
                              header=0, delimiter=",")

    df_dict = {
        "DowJones": DowJones_pd['Adj Close'],
        "Nasdaq": Nasdaq_pd['Adj Close'],
        "Russell2000": Russell2000_pd['Adj Close'],
        "SP500": SP500_pd['Adj Close']
    }

    for (name, df) in df_dict.items():
        df_dict[name] = np.asarray(df)

    data_df = pd.DataFrame(df_dict, index = DowJones_pd.Date)
    data_df.sort_index(inplace = True )
    return data_df
