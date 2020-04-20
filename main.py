#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vincent Roy @ Arthur Claude
"""

from data import load_data
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
sns.set_palette("PuBuGn_d")

def visualise(df):
    df.plot()
    plt.xticks(rotation=20)



if __name__ == "__main__":
    df = load_data()
    visualise(df)

