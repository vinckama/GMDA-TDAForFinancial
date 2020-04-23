#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vincent Roy & Arthur Claude
"""

import argparse
import sys


class DatasetCommand(object):
    def __init__(self, df, df_log):
        self.df = df
        self.df_log = df_log

        parser = argparse.ArgumentParser(
            description='Access to the dataset',
            usage=''''manage.py dataset <subcommand>'

Subcommands are:
   visualise   visualise the dataset
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

    def visualise(self):
        parser = argparse.ArgumentParser(
            description='visualise the dataset')
        # prefixing the argument with -- means it's optional
        parser.add_argument('--log', help='plot log ratio graphs',
                            action='store_true')

        args = parser.parse_args(sys.argv[3:])
        if args.log:
            self.df_log.visualise_subplots()
        else:
            self.df.visualise()


