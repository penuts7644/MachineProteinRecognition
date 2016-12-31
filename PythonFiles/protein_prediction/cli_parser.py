#! /usr/bin/env python3

"""
Author: Wout van Helvoirt
Build: January 30th, 2017
Version: 1.0

Usage:
This module needs to be imported. This class can parse commandline arguments given by user.

Input:
- A list with options necessary for parsing of commandline arguments.
"""

import optparse
import sys


class CliParser:
    def __init__(self,
                 option_list):
        self.options_list = option_list
        self.arguments = self._parse_arguments()

    def _check_input(self,
                     parser,
                     user_args):
        """
        This function checks if user has given input.
        Input: Dict containing user arguments and the parser object.
        Output: Returns the user arguments (modified if necessary).
        """

        # Check if input directory is given by user.
        for option in self.options_list:

            # If a required option, print help.
            if option[7].startswith("[REQUIRED]") and user_args[option[1]] is None:
                parser.print_help()
                sys.exit()

            # Only if value is an integer, is it within minimum and maximum.
            if type(option[4]) is int:
                if option[5] is not None and user_args[option[1]] < option[5]:
                    user_args[option[1]] = option[4]
                if option[6] is not None and user_args[option[1]] > option[6]:
                    user_args[option[1]] = option[4]

        # Return (modified) arguments.
        return user_args

    def _parse_arguments(self):
        """
        This function parses information from the commandline.
        Input: -, uses options list to create options and arguments.
        Output: Dictionary containing parsed arguments from user based on options.
        """

        # Add options specified in the option list to the OptionParser.
        parser = optparse.OptionParser(add_help_option=True)
        for option in self.options_list:
            parser.add_option("-" + option[0],
                              "--" + option[1],
                              action=option[2],
                              type=option[3],
                              default=option[4],
                              help=option[7])
        opts, args = parser.parse_args()
        return self._check_input(parser=parser,
                                 user_args=vars(opts))


def main():
    """
    Main function when program ran through terminal.
    """
    print(__doc__)


if __name__ == "__main__":
    main()
