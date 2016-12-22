#! /bin/sh

# Bash script that uses the first column of a cull pdb file to get the corresponding pdb files from given directory.
# W. van Helvoirt, 2016

# Check if arguments given by user.
if [ -z $1 ]
then
    echo "No input file path given."
    exit 1
elif [ -z $2 ]
then
    echo "No pdb location folder path given."
    exit 1
fi

# Create directory for saving files.
mkdir -p "${0%/*}/pdb_files"

# For each line in given file, split on ' '.
while IFS=' ' read -ra line

# Search for identifier code in given directory recursively.
do
    size="${#line[0]}"
    first="${line[0]:0:$size-1}"
    last="${line[0]:$size-1}"

    # Copy file to created directory.
    find "$2" -name "${first,,}${last^^}.pdb" -exec cp "{}" "${0%/*}/pdb_files" \;
    echo "Copied PDB with identifier "${first,,}${last^^}" to "${0%/*}/pdb_files""
done < "$1"
