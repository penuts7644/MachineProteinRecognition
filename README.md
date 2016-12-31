# Machine Protein Recognition

Author: Wout van Helvoirt

Build: December 14th, 2016

Version: 1.0

This Python module was build using Python version 3.5.2 and is capable to do four things. It can process the desired pdb files to be used for training. Add more diversity to the training files by slightly modifying the processed pdb files. Train a model using layers given by a configuration file and finally predict new data using the created model.
The ExampleFiles directory contains example configuration files to use for setting up the model, some test pdb files, a bash file to run the training program via a slurm manager and a bash file to collect the pdb id's from a cullpdb file.

### Getting set up

Make sure that Python 3.5.2 is installed with the following Python modules installed:
* BioPython module (version 1.68 has been used for development)
* Pandas module (version 0.19.1 has been used for development)
* TensorFlow module (version r0.10/r0.12 has been used for development)
* NumPy module (version 1.11.2 has been used for development)
* scikit-learn module (version 0.18.1 has been used for development)
* h5py module (version 2.6.0 has been used for development)
* Keras module (version 1.1.2 has been used for development)

To achieve the best performance possible, it is recommended to have a GPU installed. This should speed up the training process drastically.

### How to use this application
For more specific information on how to run the programs from start to finish, please read the `PythonFiles/README.md` file. [HERE](https://github.com/penuts7644/MachineProteinRecognition/tree/master/PythonFiles "README")

Want to know more about the configuration of the model? Please read the `ExampleFiles/README.md` file. [HERE](https://github.com/penuts7644/MachineProteinRecognition/tree/master/ExampleFiles "README")

### My use case & future ideas
My target was to train a model that could recognise real protein and fake protein. Using multiple model configurations, the model should be trained on multichannel contact matrices (with the additional amino acid interaction information).

Future ideas could be to write a custom TensorFlow operator and input producer for feeding the data files instead of reading it completely into memory. Because of this, multi-GPU system can't be used at full potential and only use one GPU instead.
