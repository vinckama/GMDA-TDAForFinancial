#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vincent Roy & Arthur Claude
"""

import argparse
import sys
from persistence import Norm


class NormCommand(object):
    def __init__(self, df, df_log):
        self.df = df
        self.df_log = df_log
        self.norm = Norm(self.df_log)

        parser = argparse.ArgumentParser(
            description='Access to the norm of the persistence',
            usage=''''manage.py norm <subcommand>'

Subcommands are:
   visualise     plot the norm graph
   get           get the norm
   crash_stats   get and plot statistics on crashs
   clean         clean the hidden working database
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
            description='compute norm')
        # prefixing the argument with -- means it's optional
        parser.add_argument(
            '-w_size',
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
        parser.add_argument('--save', 
                            help='location where you want to save graphs',
                            default='')
        args = parser.parse_args(sys.argv[3:])
        self.norm.visualise(int(args.w_size), args.start_date,  args.end_date, 
                            args.save)

    def get(self):
        parser = self.parse()
        args = parser.parse_args(sys.argv[3:])
        self.norm(int(args.w_size), args.start_date,  args.end_date)
    
    def crash_stats(self):
        parser = argparse.ArgumentParser(
            description = 'get and plot statistics on crashs')
        # prefixing the argument with -- means it's optional
        parser.add_argument(
            '-w_size',
            help = 'size of the windows for the landscapes computations',
            required = True)

        parser.add_argument(
            '-year',
            help='year of the crash',
            choices = ['2000', '2008'],
            required = True)
        
        parser.add_argument('--test', help='plot statistics',
                    action='store_true')
        
        parser.add_argument('--plot', help='compute mann kendall test',
                    action='store_true')
        
        parser.add_argument('--save', 
                            help='location where you want to save graphs',
                            default='')
        
        args = parser.parse_args(sys.argv[3:])
        self.norm.crash_stats(int(args.w_size), args.year, args.test, args.plot,
                              args.save)

    def clean(self):
        self.norm.clean_dataset()
