#! /usr/bin/env python3

"""
Author: Wout van Helvoirt
Build: January 30th, 2017
Version: 1.0

Usage:
This module needs to be imported. It is able to parse a pdb file and convert it to a distance map, contact map or
modified contact map with extra amino acid information. The input pdb files are split over multiple processes to
increase performance. Each created matrix is written as csv file to the file system with a csv like structure.

Requirements:
In order to use this module, please make sure that you have installed numpy, biopython and the text_file_writer of this
package.

Input:
- The output directory to write the created files to.
- Separator used for parsing the csv data from the input files.
- Comment character to use for header line.
- Header line to use for all the files.
- If a contact map should be made.
- The cut off value to use when converting distance matrices to contact matrices. Only used when contact maps are made.
- If additional amino acid information should be added to the contact map.
"""

import os
import sys
import numpy
from multiprocessing import Pool

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import three_to_one
from Bio.PDB.Polypeptide import is_aa

import protein_prediction as pp


class PdbToMatrixParser:
    def __init__(self,
                 output_dir,
                 separator,
                 comment_char,
                 header_line,
                 contact_matrix,
                 cutoff_value,
                 multichannel):
        self.output_dir = output_dir
        self.separator = separator
        self.comment_char = comment_char
        self.header_line = header_line
        self.contact_matrix = contact_matrix
        self.cutoff_value = cutoff_value
        self.multichannel = multichannel
        self.pdb_chain = None
        self.amino_acid_id = dict(A=1, R=2, N=3, D=4, C=5, E=6, Q=7, G=8, H=9, I=10,
                                  L=11, K=12, M=13, F=14, P=15, S=16, T=17, W=18, Y=19, V=20,
                                  X=21)

    def _add_amino_acid_information(self,
                                    matrix):
        """
        This function adds amino acid interaction information to the contact matrix.
        Input: The contact matrix to modify, uses the amino acid conversion dictionary and pdb model.
        Output: The modified version of the contact matrix.
        """

        # Build the peptide sequence for each residue in the chain.
        peptide_sequence = ""
        for residue in self.pdb_chain:
            if is_aa(residue.get_resname(), standard=True):
                peptide_sequence += three_to_one(residue.get_resname())
            else:
                peptide_sequence += "X"

        # For each combination amino acids, add there binary to the contact matrix.
        for i, ver_amino in enumerate(peptide_sequence):
            for j, hor_amino in enumerate(peptide_sequence):

                # Use the contact map value as first character, followed by corresponding amino acid i (ver)
                # value, followed by corresponding amino acid j (hor) value. Total str size incl. sep. 5to7
                matrix[i][j] = " ".join([str(matrix[i][j]),
                                         str(self.amino_acid_id[ver_amino]),
                                         str(self.amino_acid_id[hor_amino])])

        # Return the first modified contact matrix and end the loop.
        return matrix

    def _parse_pdb(self,
                   file_path):
        """
        This function parses a pdb file using BioPython's PDB module.
        Input: The pdb file path to parse.
        Output: -, sets the parsed pdb model.
        """

        # Get the identifier code form the pdb file name (first four characters).
        file_code = os.path.basename(file_path).split(".")[0][:4].upper()
        file_chain = os.path.basename(file_path).split(".")[0][-1:].upper()

        # Try to parse the pdb file and set the model.
        structure = PDBParser(PERMISSIVE=False,
                              QUIET=False).get_structure(file_code,
                                                         file_path)

        # For each model use the first one that has the chain specified in filename inside.
        for model in structure.get_models():

            # For each chain in the model, use the chain from the filename.
            for chain in model.get_chains():
                if chain.get_id() == file_chain:

                    # Remove all hetero-atoms from residues list, only keeping amino acid residues.
                    for residue in list(chain.get_residues()):
                        identifier = residue.id
                        if identifier[0] != " ":
                            chain.detach_child(identifier)

                    # Finally set the modified chain.
                    self.pdb_chain = chain
                    return

    @staticmethod
    def _calc_residue_distance(residue_one,
                               residue_two):
        """
        This function calculates the distance between two residues.
        Input: Both of the residues.
        Output: The calculated distance between the two given residues.
        """

        # Try to calculate the distance of the residues and return the value.
        try:
            diff_vector = residue_one["CA"].coord - residue_two["CA"].coord
            return numpy.sqrt(numpy.sum(diff_vector * diff_vector))

        # If 'CA' not available, return zero.
        except KeyError:
            return 0

    def _calc_distance_matrix(self,
                              file_path):
        """
        This function calculates the distance matrix from, using the chain specified in the filename.
        Input: The pdb file path.
        Output: A numpy array matrix containing the distances.
        """

        # Parse the pdb file and get the chain identifier.
        self._parse_pdb(file_path=file_path)

        # Make a numpy matrix with zeros using the chain identifier lengths.
        distance_matrix = numpy.zeros((len(self.pdb_chain), len(self.pdb_chain)),
                                      dtype="float")

        # For each position in the matrix, calculate the distance and add it to the matrix.
        for r, residue_one in enumerate(self.pdb_chain):
            for c, residue_two in enumerate(self.pdb_chain):
                distance_matrix[r][c] = self._calc_residue_distance(residue_one=residue_one,
                                                                    residue_two=residue_two)

        # Return the first distance matrix that could be created and end the loop.
        return distance_matrix

    def _process_files(self,
                       files_list_part):
        """
        This function processes each file using multiprocessing. I writes out a distance matrix, contact matrix or
        modified contact matrix to the output directory.
        Input: A part of the files list, uses cutoff value, if contact matrix should be made and if additional amino
               acid information should be added to the contact matrix.
        Output: -, writes the created numpy array matrix to a file.
        """

        # Define file writer and process each file in the files list.
        writer = pp.ListToFileWriter("csv",
                                     self.separator)
        for file_path in files_list_part:
            try:

                # Calculate distance matrix.
                matrix = self._calc_distance_matrix(file_path=file_path)

                # If contact matrix requested, convert distance matrix.
                if self.contact_matrix:

                    # Calculate contact matrix.
                    matrix = (matrix < float(self.cutoff_value)).astype("int")

                # Add amino acid information if requested.
                if self.multichannel:
                    matrix = self._add_amino_acid_information(matrix=matrix.astype("object"))

                # Write the matrix to a csv file.
                writer.write(self.output_dir,
                             str(os.path.basename(file_path).split(".")[0]),
                             matrix,
                             str(self.comment_char + " " + self.header_line))
            except ValueError:
                print("Warning: problem with PDB file, skipping file " + os.path.basename(file_path),
                      file=sys.stderr)

    def calculate_matrix(self,
                         files_list):
        """
        This function is manages processes. Each file in the files list is processed by a process. It uses all of the
        CPU' available on the system. (os.cpu_count function)
        Input: The files list, cutoff value, output directory, amount of threads, if contact matrix is wanted and if
               extra amino acid information should be added to the contact matrices.
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
