# Machine Protein Recognition

Author: Wout van Helvoirt

Build: January 30th, 2017

Version: 1.0

This Python module was build using Python version 3.5.2 and is capable to do five things. It can process the desired pdb files to be used for training. Add more diversity to the training files by slightly modifying the processed pdb files. Train a model using layers given by a configuration file, resume training with a previously created model and finally predict new data using the created model.

### 1. Generating the input data
```
-i, --input_dir,          [REQUIRED] Input path directory that contains the pdb files to process.
-o, --output_dir,         [REQUIRED] Output path directory to write the processed csv files to.
-l, --header_line,        [REQUIRED] The header line to place at the top of each generated file.
-c, --cutoff_value,       The cutoff value to be used when converting protein distance maps to protein contact maps.
                          Default integer 8
-C, --contact_files,      Do you want to generate contact matrices? If given, each distance matrix will be converted using
                          the value of --cutoff_value. Default boolean False.
-M, --multichannel_files, Do you want the matrices to have multiple channels? If given, the matrices, contact or not,
                          gain additional amino acid interaction information. The lines of the matrices will be three
                          times longer. Default boolean False.
```
This Python program (`input_data_creator.py`) requires the user to specify a directory containing pdb files. The pdb files located in the directory (and directories below it) will be processed using BioPython and written as csv files (separated by a white space) to the user given output directory. By default, this program will generate distance matrices from the pdb files. With `-C, --contact_files` the program will generate contact matrices using the cutoff value given by default or by the user with command `-c, --cutoff_value`. You also have the option to add addition amino interaction information to the output matrix files with `-M, --multichannel_files`. This setting will make the the output matrices horizontal lines three times longer using the same format csv files, but with amino acid `i` and amino acid `j` in between each distance/contact value. Amino acid `i` represents the vertical aligned sequence from top-left to bottom-left while amino acid `j` represents the horizontal aligned sequence from top-left to top-right. A required header line will be added to all of the files and will be used as class for training.

#### Example command:
```
python3 input_data_creator.py -i /location/to/pdb/files/ -o /location/of/output/files/ -l real_protein -c 8 -C -M
```
This command will make contact matrices using a cutoff value of 8 and adds additional amino acid interaction information to each file. Each file will have a header line of `real_protein` that specifies the class used for training later on.

#### Python requirements:
* BioPython module (version 1.68 has been used for development)
* It uses multiprocessing to automatically divide the workload based on the amount of processes your computer is able to run at the same time.

### 2. Generating additional modified input data
```
-i, --input_dir,           [REQUIRED] Input path directory that contains the wanted files.
-o, --output_dir,          [REQUIRED] Output path directory to write the files to.
-l, --header_line,         [REQUIRED] The header line to place at the top of each generated file.
-a, --amount_output_files, The amount of output files to be produced. Default integer 1
-m, --modify_percentage,   The percentage to modify of each of the files. Default integer 40
-M, --multichannel_files,  Do the files contain multiple channels? If given, each slice will contain three matrices:
                           contact matrix, amino acid i matrix and amino acid j matrix. Otherwise a slice will contain
                           only the contact matrix. Default boolean False.
```
To add some more diversity to the training data, a modifier (`input_data_modifier.py`) was made to change the files made by the `input_data_creator.py`. This program requires the user to specify the location of the created csv files, a location to write the modified csv files to as well as a header line for all of the modified files to be used as training class. The user can specify the amount of files to make using `-a, --amount_output_files` by which the program will make random selection of the found input files. A percentage can be given using `-m, --modify_percentage` to specify the amount of modification for each file. By default, files will be modified make taking random values from the matrix and placing them on on a position in the line that is being modified. If the user selected the `-M, --multichannel_files` option, the distance/contact value won't be changed. Instead the amino acid i and amino acid j values will be changed for the row and transposed column.

#### Example command:
```
python3 input_data_modifier.py -i /location/to/input/csv/files/ -o /location/of/output/csv/files/ -l fake_protein -m 50 -a 309 -M
```
This command will generate 309 contact matrices with amino acid information using a random selection of contact matrices with amino acid information. The selected files will be modified by 50 percent. Each file will have a header line of `fake_protein` that specifies the class used for training later on.

#### Python requirements:
* Pandas module (version 0.19.1 has been used for development)
* It uses multiprocessing to automatically divide the workload based on the amount of processes your computer is able to run at the same time.

### 3. Training the model
```
-i, --input_dir,             [REQUIRED] Input path directory that contains the wanted files.
-i, --output_dir,            [REQUIRED] Output path directory to write the model files to and TensorBoard log.
-c, --config_file_path,      [REQUIRED] The json file path to the model and compile configuration.
-b, --batch_size,            The size of the batches to be used on the train, validation and test data. Default integer 32
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
```
Now the input files are ready, lets train a model (`model_trainer.py`). Three options are required, the input directory with the csv matrices, the output directory to write the model and weights to and the model configuration json file containing the layers and compile information for the model. With `-b, --batch_size`, users can specify the batch size to use in each epoch round. This can reduce the memory usage during training by a lot. The amount of epoch rounds with the current data with `-e, --epochs`. The training rounds (`-r, --training_rounds`) specify the amount of training moments. This will split the total amount of input files to use a part of these files for each rounds. This can be use full when training on very large data sets that do not fit in memory at once. To create the three dataset (train, validation and test), the files need to be sliced in parts with `-w, --window_size`. The `-H, --hor_window_movement` and `-V, --ver_window_movement` options specify the positions to slice in the files horizontally and vertically respectively. The `-v, --validation_percentage` and `-t, --test_percentage` specify the amount of the created slices to be used for validation and test dataset respectively. The train dataset will contain al least 50 percent of all the slices. With `-M, --multichannel_files`, users can specify the type of input files. If not given, the datasets will have a shape of (total_amount_of_slices, 1, window_size, window_size). If given, the datasets will have a shape of (total_amount_of_slices, 3, window_size, window_size).

#### Example command:
```
python3 model_trainer.py -i /location/to/input/csv/files/ -o /location/to/output/directory/ -c /location/to/json/model/config/file -b 10 -e 10 -r 2 -w 100 -H 5 -V 5 -v 60 -t 60 -M
```
The model is created with the configuration file. Training is done in two rounds, each with half of the found input csv files. Within each round there will will be 10 epochs. Training will be done with batches of 10 slices. The input data is multichannel and thus contains amino acid information. The input files are sliced in matrices of 100 by 100 in size and skipping 5 steps horizontally and vertically before creating another slice. The validation and test datasets will both contain 60 percent of 1/4 of the total amount of slices. The input shape for each batch will be (10, 3, 100, 100).

#### Python requirements:
* Pandas module (version 0.19.1 has been used for development)
* TensorFlow module (version r0.10/r0.12 has been used for development)
* NumPy module (version 1.11.2 has been used for development)
* scikit-learn module (version 0.18.1 has been used for development)
* h5py module (version 2.6.0 has been used for development)
* Keras module (version 1.1.2 has been used for development)

### 4. Resume training with a previously created model
```
-i, --input_dir,             [REQUIRED] Input path directory that contains the wanted files.
-m, --model_file_path,       [REQUIRED] The file path to the model.
-w, --weights_file_path,     [REQUIRED] The file path to the pre-calculated weights.
-c, --config_file_path,      [REQUIRED] The json file path to the model and compile configuration.
-b, --batch_size,            The size of the batches used on the train, validation and test data. Default integer 32
-e, --epochs,                The number of epochs with the samples within a training rounds. Default integer 10
-r, --training_rounds,       The number of training rounds to divide the number of input files over. Default integer 1
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
```
Lets train an existing model (`model_updater.py`). Four options are required, the input directory with the csv matrices, the model file, the weights file and the model configuration json file containing compile information for the model. With `-b, --batch_size`, users can specify the batch size to use in each epoch round. This can reduce the memory usage during training by a lot. The amount of epoch rounds with the current data with `-e, --epochs`. The training rounds (`-r, --training_rounds`) specify the amount of training moments. This will split the total amount of input files to use a part of these files for each rounds. This can be use full when training on very large data sets that do not fit in memory at once. To create the three dataset (train, validation and test), the files need to be sliced in parts. The window_size is used from the model file given. The `-H, --hor_window_movement` and `-V, --ver_window_movement` options specify the positions to slice in the files horizontally and vertically respectively. The `-v, --validation_percentage` and `-t, --test_percentage` specify the amount of the created slices to be used for validation and test dataset respectively. The train dataset will contain al least 50 percent of all the slices. With `-M, --multichannel_files`, users can specify the type of input files. If not given, the datasets will have a shape of (total_amount_of_slices, 1, window_size, window_size). If given, the datasets will have a shape of (total_amount_of_slices, 3, window_size, window_size).

#### Example command:
```
python3 model_updater.py -i /location/to/input/csv/files/ -m /location/to/model/file -w /location/to/weights/file -c /location/to/json/model/config/file -b 10 -e 10 -r 2 -H 5 -V 5 -v 60 -t 60 -M
```
The model is created with the configuration file. Training is done in two rounds, each with half of the found input csv files. Within each round there will will be 10 epochs. Training will be done with batches of 10 slices. The input data is multichannel and thus contains amino acid information. The multichannel input data files are converted into one dataset using window_size found in the model file. After each slice, 5 steps horizontally and vertically before creating another slice. The validation and test datasets will both contain 60 percent of 1/4 of the total amount of slices. The input shape for each batch has this basic structure (10, 3, window_size, window_size).

#### Python requirements:
* Pandas module (version 0.19.1 has been used for development)
* TensorFlow module (version r0.10/r0.12 has been used for development)
* NumPy module (version 1.11.2 has been used for development)
* scikit-learn module (version 0.18.1 has been used for development)
* h5py module (version 2.6.0 has been used for development)
* Keras module (version 1.1.2 has been used for development)

### 5. Predicting classes of data
```
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
```
Training has been completed. The classes of new data can be predicted using the trained model (`class_predicter.py`). Four options are required, the input directory with the csv matrices, the model file, the weights file and the model configuration json file containing compile information for the model. With `-b, --batch_size`, users can specify the batch size to use when predicting to reduce memory usage. The `-H, --hor_window_movement` and `-V, --ver_window_movement` options specify the positions to slice in the files horizontally and vertically respectively. With `-M, --multichannel_files`, users can specify the type of input files. If not given, the dataset will have a shape of (total_amount_of_slices, 1, window_size, window_size). If given, the dataset will have a shape of (total_amount_of_slices, 3, window_size, window_size).

#### Example command:
```
python3 class_predicter.py -i /location/to/input/csv/files/ -m /location/to/model/file -w /location/to/weights/file -c /location/to/json/model/config/file -b 10 -H 5 -V 5 -M
```
The model is imported and the calculated weight are set. Using the configuration file the model is compiled. The multichannel input data files are converted into one dataset using window_size found in the model file. After each slice, 5 steps horizontally will be taken for the next slice and 5 steps vertically when no horizontal slices can be made from the line. Predicting is done in batches of 10.

#### Python requirements:
* Pandas module (version 0.19.1 has been used for development)
* TensorFlow module (version r0.10/r0.12 has been used for development)
* NumPy module (version 1.11.2 has been used for development)
* scikit-learn module (version 0.18.1 has been used for development)
* h5py module (version 2.6.0 has been used for development)
* Keras module (version 1.1.2 has been used for development)
