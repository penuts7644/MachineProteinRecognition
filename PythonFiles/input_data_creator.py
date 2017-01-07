#! /usr/bin/env python3

"""
Author: Wout van Helvoirt
Build: January 30th, 2017
Version: 1.0

Usage:
This program can be used for processing multiple pdb files to a contact maps like structure to be used as training data.
It uses the Python files located in the modules folder.

Input:
-i, --input_dir,          [REQUIRED] Input path directory that contains the pdb files to process.
-o, --output_dir,         [REQUIRED] Output path directory to write the processed csv files to.
-l, --header_line,        [REQUIRED] The header line to place at the top of each generated file.
-c, --cutoff_value,       The cutoff value to be used when converting protein distance matrices to protein contact
                          matrices. Default integer 8
-C, --contact_files,      Do you want to generate contact matrices? If given, each distance matrix will be converted
                          using the value of --cutoff_value. Default boolean False.
-M, --multichannel_files, Do you want the matrices to have multiple channels? If given, the matrices, contact or not,
                          gain additional amino acid interaction information. The lines of the matrices will be three
                          times longer. Default boolean False.
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
                "[REQUIRED] Input path directory that contains the pdb files to process."],
               ["o", "output_dir", "store", "string", None, None, None,
                "[REQUIRED] Output path directory to write the processed csv files to."],
               ["l", "header_line", "store", "string", None, None, None,
                "[REQUIRED] The header line to place at the top of each generated file."],
               ["c", "cutoff_value", "store", "int", 8, 1, None,
                "The cutoff value to be used when converting protein distance matrices to protein contact"
                "matrices. Default integer 8"],
               ["C", "contact_files", "store_true", None, False, None, None,
                "Do you want to generate contact matrices? If given, each distance matrix will be converted using "
                "the value of --cutoff_value. Default boolean False."],
               ["M", "multichannel_files", "store_true", None, False, None, None,
                "Do you want the matrices to have multiple channels? If given, the matrices, contact or not, "
                "gain additional amino acid interaction information. The lines of the matrices will be three "
                "times longer. Default boolean False."]]
    parse = pp.CliParser(option_list=options)

    # Set pattern and search for files in given directory.
    searcher = pp.FileSearcher(file_type="pdb",
                               amount_of_partitions=os.cpu_count())
    files_list = searcher.search_dir(input_dir=parse.arguments["input_dir"])

    # Set the pdb converter and process pdb files.
    pdb_converter = pp.PdbToMatrixParser(output_dir=parse.arguments["output_dir"],
                                         separator=" ",
                                         comment_char="#",
                                         header_line=parse.arguments["header_line"],
                                         contact_matrix=parse.arguments["contact_files"],
                                         cutoff_value=parse.arguments["cutoff_value"],
                                         multichannel=parse.arguments["multichannel_files"])
    pdb_converter.calculate_matrix(files_list=files_list)


if __name__ == "__main__":
    main()
