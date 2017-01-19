#! /usr/bin/env python3

"""
Author: Wout van Helvoirt
Build: January 19th, 2017
Version: 1.0

Usage:
This program can be used for training a model with protein contact/distance matrices as data set. It reads labels and
matrix data from the csv data files and creates label and data batches for the training, validation and testing. It
trains the model using the train and validation data and tests the model using the test data. At last, the model and
model weights will be written to disk as JSON and HDF5 file respectively. It uses the Python files located in the
modules folder.

Input:
-i, --input_dir,             [REQUIRED] Input path directory that contains the wanted files.
-i, --output_dir,            [REQUIRED] Output path directory to write the model files to and TensorBoard log.
-c, --config_file_path,      [REQUIRED] The json file path to the model and compile configuration.
-b, --batch_size,            The size of the batches used on the train, validation and test data. Default integer 32
-e, --epochs,                The number of epochs with the samples within a training rounds. Default integer 10
-r, --training_rounds,       The number of training rounds to divide the number of input files over. Default integer 1
-w, --window_size,           The window size to be used for slicing the input files. Default integer 50
-H, --hor_window_movement,   The amount of horizontal steps the window should make for a new slice of the input file.
                             Default integer 1
-V, --ver_window_movement,   Same as --hor_window_movement option, but than for vertical movement. Default integer 1
-v, --validation_percentage, The percentage to take from one-fourth of the total amount of input files. The outcome is
                             rounded down and used to define the amount of input files for the validation set. Leftover
                             amount will be added to the train set. The train set will always contain at least 50
                             percent of the total amount of input files given. Default int 80
-t, --test_percentage,       Same as --validation_percentage option, but then to make the test set. Default int 80
-M, --multichannel_files,    Do the files contain multiple channels? If given, each slice will contain three matrices:
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
                "[REQUIRED] Input path directory that contains the wanted files."],
               ["o", "output_dir", "store", "string", None, None, None,
                "[REQUIRED] Output path directory to write the model files to and TensorBoard log."],
               ["c", "config_file_path", "store", "string", None, None, None,
                "[REQUIRED] The json file path to the model and compile configuration."],
               ["b", "batch_size", "store", "int", 32, 1, None,
                "The size of the batches used on the train, validation and test data. Default integer 32"],
               ["e", "epochs", "store", "int", 10, 1, None,
                "The number of epochs with the samples within a training rounds. Default integer 10"],
               ["r", "training_rounds", "store", "int", 1, 1, None,
                "The number of training rounds to divide the number of input files over. Default integer 1"],
               ["w", "window_size", "store", "int", 50, 10, None,
                "The window size to be used for slicing the input files. Default integer 50"],
               ["H", "hor_window_movement", "store", "int", 1, 1, None,
                "The number of horizontal steps the window should make before taking a new slice of the input file. "
                "Default integer 1"],
               ["V", "ver_window_movement", "store", "int", 1, 1, None,
                "Same as --hor_window_movement option, but than for vertical movement. Default integer 1"],
               ["v", "validation_percentage", "store", "int", 80, 0, 100,
                "The percentage to take from one-fourth of the total amount of input files. The outcome is "
                "rounded down and used to define the amount of input files for the validation set. Leftover "
                "amount will be added to the train set. The train set will always contain at least 50 "
                "percent of the total amount of input files given. Default int 80"],
               ["t", "test_percentage", "store", "int", 80, 0, 100,
                "Same as --validation_percentage option, but then to make the test set. Default int 80"],
               ["M", "multichannel_files", "store_true", None, False, None, None,
                "Do the files contain multiple channels? If given, each slice will contain three matrices: "
                "contact matrix, amino acid i matrix and amino acid j matrix. Otherwise a slice will contain "
                "only the contact matrix. Default boolean False."]]
    parse = pp.CliParser(option_list=options)

    # Set pattern and search for files in given directory.
    searcher = pp.FileSearcher(file_type="csv",
                               amount_of_partitions=parse.arguments["training_rounds"])
    files_list = searcher.search_dir(input_dir=parse.arguments["input_dir"])

    # Set dataset creation requirements.
    dataset_creator = pp.CsvToDatasetParser(window_size=parse.arguments["window_size"],
                                            hor_window=parse.arguments["hor_window_movement"],
                                            ver_window=parse.arguments["ver_window_movement"],
                                            validation_size=parse.arguments["validation_percentage"],
                                            test_size=parse.arguments["test_percentage"],
                                            separator=" ",
                                            comment_char="#",
                                            multichannel=parse.arguments["multichannel_files"])

    # Initialize the model container.
    model_container = pp.ModelContainer()

    # Make the model using the configurations.
    model_container.create_model(config_file=parse.arguments["config_file_path"],
                                 window_size=parse.arguments["window_size"],
                                 multichannel=parse.arguments["multichannel_files"])

    # Generate the dataset and train the model for each part in the files list.
    for i in range(len(files_list)):
        print("Training round {0}/{1}".format(i + 1, parse.arguments["training_rounds"]))
        train_set, \
            validation_set, \
            test_set = dataset_creator.create_datasets(files_list=files_list[i])

        # Give error message if the train, validation or test sets contain zero items.
        if len(train_set[0]) > 0 or len(validation_set[0]) > 0 or len(test_set[0]) > 0:

            # Train the model with the current dataset.
            model_container.train_model(train_set=train_set,
                                        validation_set=validation_set,
                                        num_epochs=parse.arguments["epochs"],
                                        batch_size=parse.arguments["batch_size"],
                                        output_dir=parse.arguments["output_dir"])

            # Test the model accuracy.
            model_container.test_model(test_set=test_set,
                                       batch_size=parse.arguments["batch_size"])
        else:
            print("Error: Train ({0}), validation ({1}) or test ({2}) set can't contain 0 samples."
                  .format(len(train_set[0]),
                          len(validation_set[0]),
                          len(test_set[0])),
                  file=sys.stderr)
            sys.exit()

    # Write the model to a file in JSON format and weight in HDF5.
    model_container.write(output_dir=parse.arguments["output_dir"],
                          file_name="protein_prediction")


if __name__ == "__main__":
    main()
