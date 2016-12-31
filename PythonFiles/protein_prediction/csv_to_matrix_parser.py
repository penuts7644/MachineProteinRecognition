#! /usr/bin/env python3

"""
Author: Wout van Helvoirt
Build: January 30th, 2017
Version: 1.0

Usage:
This module needs to be imported. It is able to modify the contents of a csv file.

Requirements:
In order to use this module, please make sure that you have installed pandas and the text_file_writer of this package.

Input:
- The percentage of the file to modify.
- The output directory to write the created files to.
- Separator used for parsing the csv data from the input files.
- Comment character to use for avoid parsing the csv label data from the input file.
- Header line to use for all the files.
- If the files are multichannel or not.
"""

import random
import math
import os
from multiprocessing import Pool

import pandas

import protein_prediction as pp


class CsvToMatrixParser:
    def __init__(self,
                 modify_percentage,
                 output_dir,
                 separator,
                 comment_char,
                 header_line,
                 multichannel):
        self.modify_percentage = modify_percentage
        self.output_dir = output_dir
        self.separator = separator
        self.comment_char = comment_char
        self.header_line = header_line
        self.multichannel = multichannel
        self.amino_acid_id = dict(A=1, R=2, N=3, D=4, C=5, E=6, Q=7, G=8, H=9, I=10,
                                  L=11, K=12, M=13, F=14, P=15, S=16, T=17, W=18, Y=19, V=20,
                                  X=21)

    def _read_matrix(self,
                     file_path):
        """
        This function reads the data from a whitespace separated CSV file.
        Input: The file path of the to be read file, uses separator and comment character.
        Output: Returns numpy array with the data of the file.
        """

        # Load in the csv data and filename as numpy array, skip the commented lines.
        return pandas.read_csv(file_path,
                               header=None,
                               sep=self.separator,
                               comment=self.comment_char).as_matrix()

    def _get_percentage_sample(self,
                               matrix_size):
        """
        This function makes a sample of line numbers to modify based on given percentage.
        Input: The matrix size, uses percentage to modify from the file.
        Output: A random sample list with line indices.
        """

        # Make the sample list based on the amount of lines in the matrix and percentage to modify.
        return random.sample(range(matrix_size),
                             math.floor((matrix_size * self.modify_percentage) / 100))

    def _modify_singlechannel(self,
                              matrix):
        """
        This function modifies a line distance/contact value as well as it's transposed version.
        Input: The matrix, uses percentage of the matrix to modify.
        Output: Returns the modified matrix.
        """

        # Set step_size and get a sample with lines to modify.
        step_size = math.floor(len(matrix[0]) / self.modify_percentage)
        lines_sample = self._get_percentage_sample(matrix_size=len(matrix))

        # For each line, modify the line.
        for line_index in lines_sample:

            # For each three values in the line, modify amino acid i.
            for j in range(0, int(len(matrix[line_index])), step_size):

                # Pick random location in the line to use as new value.
                if line_index != 0:
                    random_new_value = random.randint(0, line_index - 1)

                    # Set the randomly selected value on the position.
                    matrix[line_index][j] = matrix[line_index][random_new_value]

                    # Do the same for the transposed value in the matrix (slightly different location).
                    matrix[j][line_index] = matrix[random_new_value][line_index]

        # return the modified matrix.
        return matrix

    def _modify_multichannel(self,
                             matrix):
        """
        This function modifies a line amino acid interaction as well as it's transposed version.
        Input: The matrix.
        Output: Returns the modified matrix.
        """

        # Get a sample with lines to modify.
        lines_sample = self._get_percentage_sample(matrix_size=len(matrix))

        # For each line, modify the line.
        for line_index in lines_sample:

            # Try to get the amino acid used in the line (position i), select a random amino acid from the dictionary.
            original_amino_acid = matrix[line_index][1]
            new_amino_acid = random.choice([v for k, v in self.amino_acid_id.items()
                                            if v != original_amino_acid])

            # For each three values in the line, modify amino acid i.
            for j in range(int(len(matrix[line_index]) / 3)):

                # Set the randomly selected value on the amino acid i position.
                matrix[line_index][(j * 3) + 1] = new_amino_acid

                # Do the same for the transposed value in the matrix (slightly different location).
                matrix[j][(line_index * 3) + 2] = new_amino_acid

        # return the modified matrix.
        return matrix

    def _process_files(self,
                       files_list_part):
        """
        This function processes each file using multiprocessing. I modifies the matrix from the contents of the file and
        writes the modified matrix as csv file to the output directory.
        Input: A part of the files list, uses the percentage of the file to modify, output directory and if the files
               are multichannel.
        Output: -, writes the created numpy array matrix to a file.
        """

        # Define file writer and process each file in the files list.
        writer = pp.ListToFileWriter("csv",
                                     self.separator)
        for file_path in files_list_part:

            # Read in the matrix from the data file.
            matrix = self._read_matrix(file_path=file_path)

            # If multichannel is true, modify amino acid interactions.
            if self.multichannel:
                self._modify_multichannel(matrix=matrix)
            else:
                self._modify_singlechannel(matrix=matrix)

            # Write the matrix to a csv file.
            writer.write(self.output_dir,
                         str(os.path.basename(file_path).split(".")[0]),
                         matrix,
                         str(self.comment_char + " " + self.header_line))

    def create_matrix(self,
                      files_list):
        """
        This function is manages processes. Each file in the files list is processed by a process. It uses all of the
        CPU' available on the system. (os.cpu_count function)
        Input: The files list.
        Output: -, makes each process write a output file.
        """

        # Divide the files list over multiple processes to process simultaneously.
        with Pool(processes=None) as p:
            p.map(self._process_files,
                  files_list)


def main():
    """
    Main function when program ran through terminal.
    """
    print(__doc__)


if __name__ == "__main__":
    main()
