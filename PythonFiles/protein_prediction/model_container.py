#! /usr/bin/env python3

"""
Author: Wout van Helvoirt
Build: January 30th, 2017
Version: 1.0

Usage:
This module needs to be imported. It is able to create (train), test, write and load models for usage. It automatically
encodes labels from string to categorical when a model is trained or tested and decodes labels when data is predicted.
When training the model, TensorBoard log is also stored in given directory.

Requirements:
In order to use this module, please make sure that you have installed tensorflow, numpy, sklearn, h5py and keras.
"""

import os
import sys
import numpy
import json

from keras.models import Sequential, model_from_json
from keras.layers import *
from keras.optimizers import *
from keras.utils.np_utils import categorical_probas_to_classes
from keras.callbacks import TensorBoard
from sklearn.preprocessing import LabelEncoder


class ModelContainer:
    def __init__(self):
        self.model = None

    @staticmethod
    def _encode_label(non_encoded_list):
        """
        This function encodes the string labels for categorical use.
        Input: A list with labels to encode.
        Output: A list of the same size as input with encoded labels.
        """

        # Set the label encoder.
        encoder = LabelEncoder()

        # Encode the labels, making them categorical.
        encoder.fit(non_encoded_list)
        encoded_list = encoder.transform(non_encoded_list)

        # Return the encoded label lists
        return encoded_list

    @staticmethod
    def _decode_label(encoded_list):
        """
        This function decodes binary/categorical labels.
        Input: A list with labels to decode and original labels.
        Output: A list of the same size as input with decoded labels.
        """

        try:

            # Set the label encoder.
            encoder = LabelEncoder()

            # Encode the labels, making them binary integer.
            encoder.fit(encoded_list[0])
            decoded_list = encoder.inverse_transform(categorical_probas_to_classes(encoded_list[1]))

            # Return the encoded label list
            return decoded_list

        except ValueError:
            print("Error: Amount of different labels to decode differs from the amount of different original labels. "
                  "Include more input files in the training round by using a smaller window size.",
                  file=sys.stderr)
            sys.exit()

    def _set_model_layers(self,
                          layer_config,
                          window_size,
                          multichannel):
        """
        This function set the layers of the model.
        Input: Dictionary containing layer information for the model, window size and if input files are multichannel.
        Output: -, sets the layers in the model.
        """

        # For each layer given, build a command and try to add it to the model.
        for i in range(len(layer_config)):
            layer_command = layer_config[i]["class_name"] + "("

            # Add configuration values from the current layer to the command.
            for key, value in layer_config[i]["config"].items():
                layer_command += key + "=" + str(value) + ","

            # Add the input shape argument to the first layer in the command.
            if i == 0:
                dimensions = int(layer_config[i]["input_shape_dimensions"])
                input_shape = [window_size] * dimensions
                if multichannel and dimensions >= 3:
                    input_shape[0] = 3
                elif not multichannel and dimensions >= 3:
                    input_shape[0] = 1
                layer_command += "input_shape=" + str(tuple(input_shape))

            # Complete the command for this layer and execute it.
            layer_command += ")"
            self.model.add(eval(layer_command))

    def _compile_model(self,
                       compile_config):
        """
        This function compiles the model.
        Input: The dictionary for compiling the model.
        Output: -, compiles the model.
        """

        # Generate the compile model command.
        # Add the start to the command.
        compile_command = "self.model.compile("

        # For each value in the config, add it to the command.
        for key, value in compile_config.items():
            compile_command += key + "=" + str(value) + ","

        # Complete the command, execute it and save the model for later use.
        compile_command += ")"
        exec(compile_command)

    def create_model(self,
                     config_file,
                     window_size,
                     multichannel):
        """
        This function creates a default model and trains it.
        Input: The config file for creating the model and compiling, window size and if input files are multichannel.
        Output: -, sets the model.
        """

        # Load the JSON file and create model if the file exists.
        if os.path.isfile(config_file):
            json_file = open(config_file, "r")
            json_config_data = json.load(json_file)
            json_file.close()

            # Set the sequential model.
            self.model = Sequential()

            # # Try to make and compile the model.
            try:
                self._set_model_layers(layer_config=json_config_data["layer_config"],
                                       window_size=window_size,
                                       multichannel=multichannel)
                self._compile_model(compile_config=json_config_data["compile_config"])
                print("Loaded configuration " + config_file)
                print(self.model.summary())

            # Except a error occurs about model/layer/init/activation etc functions.
            except NameError as e:
                print("Error: " + str(e) + ", please use the Keras documentation for available commands. The "
                      "input_shape_dimensions option should represent the amount of values in the input_shape tuple.",
                      file=sys.stderr)
                sys.exit()

        # If one of the files does not exists, report to user.
        else:
            print("Error: Given config file path is not valid.",
                  file=sys.stderr)
            sys.exit()

    @staticmethod
    def _batch_generator(batch_size,
                         data,
                         labels,
                         shuffle):
        """
        This function generates (shuffled) batches from the input data to reduce memory usage.
        Input: The batch size, data, if wanted the label data and if the indexes should be shuffled.
        Output: -, yield data and if wanted label batches.
        """

        # Random seed needs to be fixed for reproducibility.
        numpy.random.seed(7)

        # Set the counter, a sample index numpy array and the number of batches.
        counter = 0
        sample_index = numpy.arange(len(data))
        number_of_batches = numpy.ceil(len(data) / batch_size)

        # Shuffle the sample index if wanted.
        if shuffle:
            numpy.random.shuffle(sample_index)

        # While their are still batches to be made.
        while True:
            batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]

            # Create the data batches.
            data_batch = numpy.array([data[i] for i in batch_index])
            counter += 1

            # Labels been given, create labels batches and yield both.
            if labels is not None:
                label_batch = numpy.array([labels[i] for i in batch_index])
                yield data_batch, label_batch

            # No label data given, only yield data batches.
            else:
                yield data_batch
            if counter == number_of_batches:
                counter = 0
                if shuffle:
                    numpy.random.shuffle(sample_index)

    def train_model(self,
                    train_set,
                    validation_set,
                    num_epochs,
                    batch_size,
                    output_dir):
        """
        This function trains the model and encodes the train and validation labels.
        Input: The train set, validation set, training rounds, batch size and output directory.
        Output: -, trains the model and saves TensorBoard log to output directory.
        """

        # Print information about sample sizes.
        print("Training and validating on {0} and {1} samples respectively per epoch".format(len(train_set[0]),
                                                                                             len(validation_set[0])))
        # Train the model on the trainings data and use the validation data as validation.
        self.model.fit_generator(generator=self._batch_generator(batch_size=batch_size,
                                                                 data=train_set[0],
                                                                 labels=self._encode_label(train_set[1]),
                                                                 shuffle=True),
                                 nb_epoch=num_epochs,
                                 samples_per_epoch=len(train_set[0]),
                                 validation_data=self._batch_generator(batch_size=batch_size,
                                                                       data=validation_set[0],
                                                                       labels=self._encode_label(validation_set[1]),
                                                                       shuffle=True),
                                 nb_val_samples=len(validation_set[0]),
                                 callbacks=[TensorBoard(log_dir=output_dir, histogram_freq=1, write_graph=True)],
                                 verbose=1)
        print("Training and validation -- Done")

    def test_model(self,
                   test_set,
                   batch_size):
        """
        This function test a given model with test data and encodes the test labels.
        Input: Test data and batch size, uses the model to test for accuracy.
        Output: -, print information about the model.
        """

        # Test and print the trained model accuracy.
        print("Testing on {0} samples".format(len(test_set[0])), end="")
        sys.stdout.flush()

        scores = self.model.evaluate_generator(generator=self._batch_generator(batch_size=batch_size,
                                                                               data=test_set[0],
                                                                               labels=self._encode_label(test_set[1]),
                                                                               shuffle=False),
                                               val_samples=len(test_set[0]))

        # Print final results.
        print(" -- Done\n", end="")
        sys.stdout.flush()
        print("{0}: {1}%".format(self.model.metrics_names[1],
                                 scores[1] * 100))

    def predict_data(self,
                     data_set,
                     batch_size):
        """
        This function predicts the data using a created model.
        Input: The data set to predict and batch size, uses model.
        Output: -, print information about the data set prediction.
        """

        print("Predicting on {0} samples".format(len(data_set[0])), end="")
        sys.stdout.flush()

        # Calculate the predictions data.
        predictions = self.model.predict_generator(generator=self._batch_generator(batch_size=batch_size,
                                                                                   data=data_set[0],
                                                                                   labels=None,
                                                                                   shuffle=False),
                                                   val_samples=len(data_set[0]))
        print(" -- Done\n", end="")
        sys.stdout.flush()

        # BELOW FOR TESTING PURPOSES!

        # Decode the labels to string.
        decoded_predictions = self._decode_label([data_set[1],
                                                  predictions])

        correct = len([i for i, j in zip(data_set[1], decoded_predictions) if i == j])
        print("\n")
        # print("Original:\n", data_set[1])
        # print("Predicted:\n", decoded_predictions)
        print("Amount of wrongly predicted classes:\n",
              len(data_set[1]) - correct,
              "from total of {0} input slices.".format(len(data_set[1])))
        print("Accuracy of {0}%".format((correct / len(data_set[1])) * 100))
        print("\n")

    @staticmethod
    def _modify_filename(output_dir,
                         file_name,
                         file_type):
        """
        This function modifies the file name if the given file name exists.
        Input: The output directory, filename to check and file type.
        Output: The filename given, modified if necessary.
        """

        # Set original file name and count the file names tried.
        orig_file_name = file_name
        file_count = 1

        # Keep modifying the file name if it exists.
        while os.path.isfile(os.path.join(output_dir, file_name + "." + file_type)):
            file_name = str(orig_file_name) + "_" + str(file_count)
            file_count += 1

        # Return the file name when unique.
        return file_name + "." + file_type

    def write(self,
              output_dir,
              file_name):
        """
        This function writes the model to a JSON file and the weights as HDF5.
        Input: Output directory and a model/weight file name, uses the model.
        Output: -, writes the model and weights to output files.
        """

        # Create directory's recursively if not exists.
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Check if the file name is unique, modify it if necessary.
        unique_model_file_name = self._modify_filename(output_dir=output_dir,
                                                       file_name=file_name + "_model",
                                                       file_type="json")

        # Serialize model to JSON format and write it to a file.
        model_json = self.model.to_json()
        with open(os.path.join(output_dir, unique_model_file_name), "w") as json_file:
            json_file.write(model_json)
        print("File " + os.path.join(output_dir, unique_model_file_name) + " written")

        # Check if the file name is unique, modify it if necessary.
        unique_weights_file_name = self._modify_filename(output_dir=output_dir,
                                                         file_name=file_name + "_weights",
                                                         file_type="h5")

        # Serialize the calculated weights to HDF5.
        self.model.save_weights(os.path.join(output_dir, unique_weights_file_name))
        print("File " + os.path.join(output_dir, unique_weights_file_name) + " written")

    def read(self,
             model_file,
             weights_file,
             config_file):
        """
        This function create a model from a JSON file with the weights stored as HDF5.
        Input: The model file, weights file and config file for compiling the model.
        Output: -, sets the model.
        """

        # Load the JSON file and create model if the file exists.
        if os.path.isfile(model_file) or os.path.isfile(weights_file) or os.path.isfile(config_file):
            json_file = open(model_file, "r")
            json_model_config = json_file.read()
            json_file.close()
            self.model = model_from_json(json_model_config)
            print("Loaded model " + model_file)

            # Load the weights into the new model.
            self.model.load_weights(weights_file)
            print("Loaded weights " + weights_file)

            # Try compile the loaded model using given config file.
            try:
                json_file = open(config_file, "r")
                json_config_data = json.load(json_file)
                json_file.close()
                self._compile_model(compile_config=json_config_data["compile_config"])
                print("Loaded configuration " + config_file)

            # Except a error occurs about loss/optimizer functions.
            except Exception as e:
                print("Error: " + str(e) + ", please use the Keras API documentation for available commands.",
                      file=sys.stderr)
                sys.exit()

        # If one of the files does not exists, report to user.
        else:
            print("Error: Given model, weights or config file path is not valid.",
                  file=sys.stderr)
            sys.exit()


def main():
    """
    Main function when program ran through terminal.
    """
    print(__doc__)


if __name__ == "__main__":
    main()
