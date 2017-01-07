#! /usr/bin/env python3

"""
Author: Wout van Helvoirt
Build: January 30th, 2017
Version: 1.0

Usage:
This program can be used for creating fake matrices to supplement the input data set for training in TensorFlow.
It uses multiple existing protein distance/contact matrices to (csv file) to generate one fake distance/contact matrix.
It combines random line blocks from each input file and adds them randomly to a template distance/contact matrix. The
lower triangle of the matrix is transposed to the upper part of the matrix. The final matrix is validated and 'None'
values are replaced with random values in the matrix. It uses the Python files located in the modules folder.

Input:
-i, --input_dir,           [REQUIRED] Input path directory that contains the wanted files.
-o, --output_dir,          [REQUIRED] Output path directory to write the files to.
-l, --header_line,         [REQUIRED] The header line to place at the top of each generated file.
-a, --amount_output_files, The amount of output files to be produced. Default integer 1
-m, --modify_percentage,   The percentage to modify of each of the files. Default integer 40
-M, --multichannel_files,  Do the files contain multiple channels? If given, each slice will contain three matrices:
                           contact matrix, amino acid i matrix and amino acid j matrix. Otherwise a slice will contain
                           only the contact matrix. Default boolean False.
"""

import os

import protein_prediction as pp


def main():
    """
    Main function when program ran through terminal.
    """

    # Define parser and parse arguments.
    # option, extended option, action, type, default, minimum, maximum, help/required info
    options = [["i", "input_dir", "store", "string", None, None, None,
                "[REQUIRED] Input path directory that contains the wanted files."],
               ["o", "output_dir", "store", "string", None, None, None,
                "[REQUIRED] Output path directory to write the files to."],
               ["l", "header_line", "store", "string", None, None, None,
                "[REQUIRED] The header line to place at the top of each generated file."],
               ["a", "amount_output_files", "store", "int", 1, 1, None,
                "The amount of output files to be produced. Default integer 1"],
               ["m", "modify_percentage", "store", "int", 40, 0, 100,
                "The percentage to modify of each of the files. Default integer 40"],
               ["M", "multichannel_files", "store_true", None, False, None, None,
                "Do the files contain multiple channels? If given, each slice will contain three matrices: "
                "contact matrix, amino acid i matrix and amino acid j matrix. Otherwise a slice will contain "
                "only the contact matrix. Default boolean False."]]
    parse = pp.CliParser(option_list=options)

    # Set pattern and search for files in given directory.
    set_dir = pp.FileSearcher(file_type="csv",
                              amount_of_partitions=os.cpu_count())
    files_list = set_dir.search_dir(input_dir=parse.arguments["input_dir"])

    # Make a random selection of files in the input directory.
    random_files_list = set_dir.get_random_selection(files_list=files_list,
                                                     amount_files=parse.arguments["amount_output_files"])

    # Set the separator, comment character and process files.
    matrix_creator = pp.CsvToMatrixParser(modify_percentage=parse.arguments["modify_percentage"],
                                          output_dir=parse.arguments["output_dir"],
                                          separator=" ",
                                          comment_char="#",
                                          header_line=parse.arguments['header_line'],
                                          multichannel=parse.arguments["multichannel_files"])
    matrix_creator.create_matrix(files_list=random_files_list)


if __name__ == "__main__":
    main()
