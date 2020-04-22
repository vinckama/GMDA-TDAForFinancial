#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vincent Roy & Arthur Claude
"""
from math import log


def log_df(df):
    shifted_df = df.shift(-1)
    ratio_df = df / shifted_df
    ratio_df.dropna(inplace = True)
    return ratio_df.applymap(lambda x: log(x))
