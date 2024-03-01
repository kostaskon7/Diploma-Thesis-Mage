import os

# Define the file path
file_path = "/home/ubuntu/anaconda3/envs/mage/lib/python3.10/site-packages/timm/models/layers/helpers.py"

# Check if the file exists
if os.path.isfile(file_path):
    try:
        # Read the contents of the file
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Replace the specific line
        with open(file_path, 'w') as file:
            for line in lines:
                if line.strip() == "from torch._six import container_abcs":
                    file.write("import collections.abc as container_abcs\n")
                else:
                    file.write(line)
        print("File updated successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
else:
    print("File does not exist at the specified path.")