#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vincent Roy & Arthur Claude
"""
from math import log
import sys


def log_df(df):
    shifted_df = df.shift(-1)
    ratio_df = df / shifted_df
    ratio_df.dropna(inplace = True)
    return ratio_df.applymap(lambda x: log(x))


class ProgressBar:
    def __init__(self, iteration=0, total=100, decimals=1, bar_length=20,
                 prefix = 'Progress', suffix = 'Completed'):
        self.iteration = iteration
        self.total = total
        self.decimals = decimals
        self.bar_length = bar_length
        self.prefix = prefix
        self.suffix = suffix
        self.filling_chars = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉"]
        self.whole_char = "▉"
        self.empty_char = ' '
        self.begin_char = '|'
        self.end_char = '|'
        self.output_length = 0
        self.update_progress(iteration)

    def create_bar(self, iteration):
        whole_width = int(self.bar_length*iteration // self.total)
        whole_chars = self.whole_char * whole_width

        part_width = int((self.bar_length*iteration % self.total) // (self.total/8))
        part_char = self.filling_chars[part_width]

        residual_width = self.bar_length - whole_width - 1
        residual_char = self.empty_char * residual_width
        if residual_width < 0:
            part_char = ""

        bar = f"{self.begin_char}" \
              + f"{whole_chars}{part_char}{residual_char}"\
              + f"{self.end_char}"
        return bar

    def update_progress(self, iteration):
        self.iteration = iteration
        sys.stdout.write("\b" * self.output_length)

        percent = f"{100 * (iteration / float(self.total)):.{self.decimals}f}"
        bar = self.create_bar(iteration)

        output = f'{self.prefix} {bar} {percent}% {self.suffix}'

        if iteration == self.total:
            output = output + '\n'

        self.output_length = len(output)
        sys.stdout.write(output)
        sys.stdout.flush()

    def __next__(self):
        self.update_progress(iteration = self.iteration + 1)

def main():
    import time
    a = ProgressBar(total = 1000)
    for k in range(0, 1001):
        time.sleep(0.005)
        a.update_progress(k)

if __name__ =="__main__":
    main()
