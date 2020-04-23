#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Vincent Roy & Arthur Claude
"""

import argparse
import sys
from utils import verify_lib
from LoadData import DataLoader
import command

libraries = [
    'numpy',
    'pandas',
    'gudhi',
    'matplotlib'
]


class Manager(object):
    def __init__(self):

        for library in libraries:
            verify_lib(library)

        self.df, self.df_log = (DataLoader())()

        parser = argparse.ArgumentParser(
            description='Manager of the project',
            usage=''''manage.py <command>'

The most commonly used commands are:
   dataset      Access to the dataset
   landscape    Access to the landscape persistence
   norm         Access to the norm of the persistence
   bottleneck   Access to the bottleneck of the persistence
''')
        parser.add_argument('command', help='Subcommand to run')

        if len(sys.argv) < 2:
            print('No command')
            parser.print_help()
            exit(1)
        # parse_args defaults to [1:] for args, but you need to
        # exclude the rest of the args too, or validation will fail
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)

        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def dataset(self):
        command.DatasetCommand(self.df, self.df_log)

    def landscape(self):
        command.LandscapeCommand(self.df, self.df_log)

    def norm(self):
        command.NormCommand(self.df, self.df_log)

    def bottleneck(self):
        command.BottleneckCommand(self.df, self.df_log)


if __name__ == '__main__':
    Manager()
