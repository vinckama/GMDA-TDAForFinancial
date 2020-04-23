#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vincent Roy & Arthur Claude
"""

import argparse
import sys
from persistence import Bottleneck

class BottleneckCommand(object):
    def __init__(self, df, df_log):
        self.df = df
        self.df_log = df_log
        self.bottleneck = Bottleneck(self.df_log)

        parser = argparse.ArgumentParser(
            description='Access to the bottleneck of the persistence',
            usage=''''manage.py norm <subcommand>'

Subcommands are:
   visualise  plot the bottleneck graph
   get        get the bottleneck distance
''')
        parser.add_argument('command', help='Subcommand to run')

        if len(sys.argv) < 3:
            print('No command')
            parser.print_help()
            exit(1)
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[2:3])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)

        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    @staticmethod
    def parse():
        parser = argparse.ArgumentParser(
            description='visualise the dataset')
        # prefixing the argument with -- means it's optional
        parser.add_argument(
            '--w_size',
            help='size of the windows for the landscapes computations',
            required=True)

        parser.add_argument(
            '--start_date',
            help='the first day used to compute persistence graphs',
            default = None)

        parser.add_argument(
            '--end_date',
            help='the last day used to compute persistence graphs',
            default = None)
        return parser

    def visualise(self):
        parser = self.parse()
        args = parser.parse_args(sys.argv[3:])
        self.bottleneck.visualise(int(args.w_size),
                                  args.start_date,  args.end_date)

    def get(self):
        parser = self.parse()
        args = parser.parse_args(sys.argv[3:])
        self.bottleneck(int(args.w_size), args.start_date,  args.end_date)
