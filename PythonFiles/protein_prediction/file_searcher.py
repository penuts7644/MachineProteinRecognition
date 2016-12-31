#! /usr/bin/env python3

"""
Author: Wout van Helvoirt
Build: January 30th, 2017
Version: 1.0

Usage:
This module needs to be imported. It is able to search a directory recursively for files matching a specified file
extension. It can also return a random selection of the files that were found previously.

Input:
- File extension type.
"""

import os
import re
import sys
import random


class FileSearcher:
    def __init__(self,
                 file_type,
                 amount_of_partitions):
        self.file_type = file_type
        self.amount_of_partitions = amount_of_partitions

    def search_dir(self,
                   input_dir):
        """
        This function searches the given input directory for given file types.
        Input: The input directory and the amount of partition to make, uses file extension.
        Output: A partitioned list containing lists with file paths corresponding to pattern.
        """

        # If directory does not exist, give error.
        if not os.path.isdir(input_dir):
            print("Error: Given input directory is not valid",
                  file=sys.stderr)
            sys.exit()

        # Search directory recursively for files containing file extension.
        else:
            matched_files = [[] for _ in range(self.amount_of_partitions)]
            add_position = 0

            # Compile search pattern.
            file_type = re.compile(".*\." + self.file_type.split(".", -1)[-1] + "$",
                                   re.IGNORECASE)

            # For each file, add file to list if it matches pattern.
            print("Searching for input files", end="")
            sys.stdout.flush()
            for root, dirs, files in os.walk(input_dir):
                for file in filter(file_type.match, files):
                    matched_files[add_position].append(os.path.join(os.path.abspath(root),
                                                                    file))
                    if add_position == (self.amount_of_partitions - 1):
                        add_position = 0
                    else:
                        add_position += 1
            print(" -- Done\n", end="")
            sys.stdout.flush()
            return matched_files

    def get_random_selection(self,
                             files_list,
                             amount_files):
        """
        This function returns a random selection of files from the found files the input directory.
        Input: The list containing lists with file paths.
        Output: List containing randomly selected file paths.
        """

        # If the sample has a larger size than input list, makes sizes the same.
        if sum(len(i) for i in files_list) < amount_files:
            amount_files = sum(len(i) for i in files_list)

        # Make a sample list from randomly selected items in the files list.
        random_files_list = random.sample([j for i in files_list for j in i], amount_files)

        # Set the new partition list and add position.
        random_partitioned_files_list = [[] for _ in range(self.amount_of_partitions)]
        add_position = 0

        # Partition the random files list and return it.
        for file in random_files_list:
            random_partitioned_files_list[add_position].append(file)
            if add_position == (self.amount_of_partitions - 1):
                add_position = 0
            else:
                add_position += 1
        return random_partitioned_files_list


def main():
    """
    Main function when program ran through terminal.
    """
    print(__doc__)


if __name__ == "__main__":
    main()
