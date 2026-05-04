#!/bin/bash
# Prepare the project.tar.gz for CHTC submission
# Works on both macOS and Linux

echo "Preparing project.tar.gz for CHTC submission..."

# Define the files to include
FILES=(
    "cow_environment2.py"
    "dqn_learning.py"
    "utility.py"
    "animal_constants_2025.py"
    "animal_constants_OG.py"
    "animal_constants_OB.py"
    "animal_constants_UG.py"
    "animal_constants_UB.py"
)

# Create a temporary directory structure
mkdir -p project_temp/project

# Copy the files
for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        cp "$f" project_temp/project/
    else
        echo "ERROR: File $f not found!"
        rm -rf project_temp
        exit 1
    fi
done

# Create the tarball
cd project_temp
tar -czf ../project.tar.gz project/
cd ..

# Cleanup
rm -rf project_temp

if [ -f "project.tar.gz" ]; then
    echo "Successfully created project.tar.gz"
    echo "Contents:"
    tar -tzf project.tar.gz
    echo ""
    echo "Size: $(du -h project.tar.gz | cut -f1)"
else
    echo "ERROR: Failed to create project.tar.gz"
    exit 1
fi
