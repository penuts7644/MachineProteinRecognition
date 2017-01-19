#! /usr/bin/env python3

"""
Author: Wout van Helvoirt
Build: January 19th, 2017
Version: 1.0

Usage:
This module needs to be imported. It is able to parse input csv data files to train, validation and test data sets
including the labels for each data set. It slices each file into multiple parts. Each slice can contain 1 or 3 matrices
for training on distance/contact matrices or modified contact matrices (with amino acid information) respectively. The
output shape of the data (slices) is either (1, window_size, window_size) or (3, window_size, window_size).

Requirements:
In order to use this module, please make sure that you have installed pandas.

Input:
- The window size to be used for slicing the input data.
- The horizontal window movement steps between each slice.
- The vertical window movement steps between each slice.
- The validation data percentage to use from 25% of the total input.
- The test data percentage to use from 25% of the total input.
- Separator used for parsing the csv data from the input files.
- Comment character to use for parsing the csv label data from the output file.
- If the input files are multichannel or not.
"""

import math
import random
import sys

import pandas


class CsvToDatasetParser:
    def __init__(self,
                 window_size,
                 hor_window,
                 ver_window,
                 validation_size,
                 test_size,
                 separator,
                 comment_char,
                 multichannel):
        self.window_size = window_size
        self.hor_window = hor_window
        self.ver_window = ver_window
        self.validation_size = validation_size
        self.test_size = test_size
        self.separator = separator
        self.comment_char = comment_char
        self.multichannel = multichannel

    def _read_matrix(self,
                     file_path):
        """
        This function reads the data from a whitespace separated CSV file.
        Input: The file path of the to be read file, uses separator and comment character.
        Output: A numpy array with the data of the file.
        """

        # Load in the csv data, skip the commented lines and return data as matrix.
        return pandas.read_csv(file_path,
                               header=None,
                               sep=self.separator,
                               comment=self.comment_char).as_matrix()

    def _create_matrix_slices(self,
                              matrix,
                              examples):
        """
        This function creates multiple matrix slices from the given matrix, these can be single or multi channel.
        Input: The matrix to slice and examples list to fill.
        Output: Return the amount of slices made and updated examples list.
        """

        # Keep track of amount of slices produced with counter.
        slice_amount = 0

        # Use multiple range to move through the matrix.
        for i in range(0, len(matrix) - self.window_size, self.ver_window):

            # Files are multichannel, get amino acid information using steps of 3.
            if self.multichannel:
                for j in range(0, len(matrix[0]) - (self.window_size * 3), self.hor_window * 3):

                    # Get the parts for each channel.
                    matrix_1 = matrix[i:i + self.window_size, j:j + (self.window_size * 3):3]
                    matrix_2 = matrix[i:i + self.window_size, j + 1:j + 1 + (self.window_size * 3):3]
                    matrix_3 = matrix[i:i + self.window_size, j + 2:j + 1 + (self.window_size * 3):3]

                    # Append the slice containing the 3 parts to the example list.
                    examples.append([matrix_1,
                                     matrix_2,
                                     matrix_3])
                    slice_amount += 1

            # If files contain contact map only, don't use step size.
            else:
                for j in range(0, len(matrix[0]) - self.window_size, self.hor_window):

                    # Get the matrix slice.
                    matrix_1 = matrix[i:i + self.window_size, j:j + self.window_size]

                    # Append the matrix slice to the example list.
                    examples.append([matrix_1])
                    slice_amount += 1

        # Return updated examples list and the amount of slices made.
        return examples, slice_amount

    def _read_label(self,
                    file_path,
                    slice_amount,
                    labels):
        """
        This function reads in the header label from a CSV file if any.
        Input: The file path, slice amounts taken and the labels list to fill, uses separator and comment character.
        Output: Returns updated labels list.
        """

        # Get the first line in file, return it.
        with open(file_path) as file:
            line = file.readline()
            if line.startswith(self.comment_char):

                # For each slice added, add the same labels array.
                for i in range(slice_amount):

                    # Append label array to the list.
                    labels.append(line.lstrip(self.comment_char + " ").rstrip(" \n"))
                return labels

    def _create_partition(self,
                          amount_examples):
        """
        This function makes the partitioning template for separating the data.
        Input: The length of the examples list, uses validation, test size percentages
        Output: The partitioning list.
        """

        # Set the partition list.
        partition = []

        # Create a partition vector for partitioning the data later on. Set at least 50 percent as train data.
        partition += [0] * math.ceil(amount_examples / 2)

        # Set the validation and test set size, minimum of 0 percent and maximum of 25 percent of total amount of data.
        partition += [1] * math.floor(math.floor(amount_examples / 4) * (self.validation_size / 100))
        partition += [2] * math.floor(math.floor(amount_examples / 4) * (self.test_size / 100))

        # Leftover data not used as validation or test set gets added to the train set.
        partition += [0] * (amount_examples - len(partition))

        # Try to randomly shuffle the partition list to decrease bias and return.
        random.shuffle(partition)
        return partition

    @staticmethod
    def _partition_list(data,
                        partition,
                        amount_of_partitions):
        """
        This function partitions the data list, using the partition list, to create multiple output lists.
        Input: The input data list, partition list and wanted amount of partitions
        Output: Returns partitioned data lists, the output amount is based on the unique values is partition list.
        """

        # Make output list layout, it has at least the same size as the unique set of partitions.
        if len(set(partition)) > amount_of_partitions:
            amount_of_partitions = len(set(partition))
        output_list = [[] for _ in range(amount_of_partitions)]

        # For each item in the partition list, append the data item to output_list index position.
        for i in range(len(partition)):
            output_list[partition[i]].append(data[i])
        return output_list

    def create_datasets(self,
                        files_list):
        """
        This function contains the main pipeline for reading and processing input data to be used with training.
        Input: A list with files to process and the amount.
        Output: Returns the three data sets for training.
        """

        # Random seed needs to be fixed for reproducibility.
        random.seed(7)

        # Set the used lists.
        examples, labels = [], []

        # Print information about process.
        print("Processing {0} input files".format(len(files_list)), end="")
        sys.stdout.flush()

        # Do something with each file path.
        for file_path in files_list:

            # Get matrix data from file.
            matrix = self._read_matrix(file_path=file_path)

            # If matrix is correct size, make slices.
            if len(matrix) >= self.window_size:
                examples, slice_amount = self._create_matrix_slices(matrix=matrix,
                                                                    examples=examples)

                # Get the header from file as label.
                labels = self._read_label(file_path=file_path,
                                          slice_amount=slice_amount,
                                          labels=labels)

        # Create the partition for the examples list.
        partition = self._create_partition(amount_examples=len(examples))

        # Partition the input data and labels into train, validation and test sets.
        train_examples, \
            validation_examples, \
            test_examples = self._partition_list(data=examples,
                                                 partition=partition,
                                                 amount_of_partitions=3)
        train_labels, \
            validation_labels, \
            test_labels = self._partition_list(data=labels,
                                               partition=partition,
                                               amount_of_partitions=3)
        print(" -- Done\n", end="")
        sys.stdout.flush()

        # Return the data.
        return [train_examples, train_labels], \
               [validation_examples, validation_labels], \
               [test_examples, test_labels]


def main():
    """
    Main function when program ran through terminal.
    """
    print(__doc__)


if __name__ == "__main__":
    main()
