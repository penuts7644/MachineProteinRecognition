#! /usr/bin/env python3

"""
Author: Wout van Helvoirt
Build: January 30th, 2017
Version: 1.0

Usage:
This program can be used for predicting the class in a data set using a pre-calculated model and weights file.
It uses the Python files located in the modules folder.

Input:
-i, --input_dir,           [REQUIRED] Input path directory that contains the files to predict.
-m, --model_file_path,     [REQUIRED] The file path to the model.
-w, --weights_file_path,   [REQUIRED] The file path to the pre-calculated weights.
-c, --config_file_path,    [REQUIRED] The json file path to the model and compile configuration.
-b, --batch_size,          The size of the batches to be used on the prediction data. Default integer 32
-H, --hor_window_movement, The amount of horizontal steps the window, defined in the model, should make for a new slice
                           of the input file. Default integer 1
-V, --ver_window_movement, Same as --hor_window_movement option, but than for vertical movement. Default integer 1
-M, --multichannel_files,  Do the files contain multiple channels? If given, each slice will contain three matrices:
                           contact matrix, amino acid i matrix and amino acid j matrix. Otherwise a slice will contain
                           only the contact matrix. Default boolean False.
"""

import sys

import protein_prediction as pp


def main():
    """
    Main function when program ran through terminal.
    """

    # Define parser and parse arguments.
    # option, extended option, action, type, default, minimum, maximum, help/required info
    options = [["i", "input_dir", "store", "string", None, None, None,
                "[REQUIRED] Input path directory that contains the wanted files for prediction."],
               ["m", "model_file_path", "store", "string", None, None, None,
                "[REQUIRED] The file path to the model."],
               ["w", "weights_file_path", "store", "string", None, None, None,
                "[REQUIRED] The file path to the pre-calculated weights."],
               ["c", "config_file_path", "store", "string", None, None, None,
                "[REQUIRED] The json file path to the model and compile configuration."],
               ["b", "batch_size", "store", "int", 32, 1, None,
                "The size of the batches to be used on the prediction data. Default integer 32"],
               ["H", "hor_window_movement", "store", "int", 1, 1, None,
                "The amount of horizontal steps the window, defined in the model, should make for a new slice of the"
                "input file. Default integer 1"],
               ["V", "ver_window_movement", "store", "int", 1, 1, None,
                "Same as --hor_window_movement option, but than for vertical movement. Default integer 1"],
               ["M", "multichannel_files", "store_true", None, False, None, None,
                "Do the files contain multiple channels? If given, each slice will contain three matrices: "
                "contact matrix, amino acid i matrix and amino acid j matrix. Otherwise a slice will contain "
                "only the contact matrix. Default boolean False."]]
    parse = pp.CliParser(option_list=options)

    # Initialize the model container.
    model_container = pp.ModelContainer()

    # Load the model plus weights and set the compile method.
    model_container.read(model_file=parse.arguments["model_file_path"],
                         weights_file=parse.arguments["weights_file_path"],
                         config_file=parse.arguments["config_file_path"])

    # Set pattern and search for files in given directory.
    searcher = pp.FileSearcher(file_type="csv",
                               amount_of_partitions=1)
    files_list = searcher.search_dir(input_dir=parse.arguments["input_dir"])

    # Get the window size used in the model, set requirements and parse csv files.
    window_size = model_container.model.get_config()[0]["config"]["batch_input_shape"][-1]
    dataset_creator = pp.CsvToDatasetParser(window_size=window_size,
                                            hor_window=parse.arguments["hor_window_movement"],
                                            ver_window=parse.arguments["ver_window_movement"],
                                            validation_size=0,
                                            test_size=0,
                                            separator=" ",
                                            comment_char="#",
                                            multichannel=parse.arguments["multichannel_files"])
    data_set = dataset_creator.create_datasets(files_list=files_list[0])[0]

    # Give error message if the data set contains zero items.
    if len(data_set[0]) > 0:

        # Calculate the predictions for the data data.
        model_container.predict_data(data_set=data_set,
                                     batch_size=parse.arguments["batch_size"])
    else:
        print("Error: Prediction data set ({0}) can't contain 0 samples.".format(len(data_set[0])),
              file=sys.stderr)
        sys.exit()


if __name__ == "__main__":
    main()
