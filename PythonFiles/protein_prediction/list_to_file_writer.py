#! /usr/bin/env python3

"""
Author: Wout van Helvoirt
Build: December 14th, 2016
Version: 1.0

Usage:
This module needs to be imported. This module can write contents to a file on the file system.

Input:
- File extension type.
- Separator used for writing the output file.
"""

import os


class ListToFileWriter:
    def __init__(self,
                 file_type,
                 separator):
        self.file_type = file_type
        self.separator = separator

    def _modify_filename(self,
                         output_dir,
                         file_name):
        """
        This function modifies the file name if the given file name exists.
        Input: The output directory and filename to check.
        Output: The filename given, modified if necessary.
        """

        # Set original file name and count the file names tried.
        orig_file_name = file_name
        file_count = 1

        # Keep modifying the file name if it exists.
        while os.path.isfile(os.path.join(output_dir, file_name + "." + self.file_type)):
            file_name = str(orig_file_name) + "_" + str(file_count)
            file_count += 1

        # Return the file name when unique.
        return file_name

    def write(self,
              output_dir,
              file_name,
              contents,
              header):
        """
        This function writes contents to a file.
        Input: The output directory and file name, the contents to be written and an optional header line.
        Output: -, writes the contents to a file.
        """

        # Create directory's recursively if not exists.
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Check if the file name is unique, modify it if necessary.
        unique_file_name = self._modify_filename(output_dir=output_dir,
                                                 file_name=file_name)

        # Write contents to file.
        with open(os.path.join(output_dir, unique_file_name + "." + self.file_type), mode="wt", encoding="utf-8") as f:

            # Write the header, make the output string, remove last newline in string.
            f.write(header + "\n")
            output_string = ""
            for items in contents:
                output_string += str(self.separator.join([str(item) for item in items])) + "\n"
            f.write(output_string)
            print("File " + os.path.join(output_dir, unique_file_name + "." + self.file_type) + " written")


def main():
    """
    Main function when program ran through terminal.
    """
    print(__doc__)


if __name__ == "__main__":
    main()
