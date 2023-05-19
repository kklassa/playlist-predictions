import fileinput

file_path = "./example_data.txt"

for line in fileinput.input(file_path, inplace=True):
    updated_line = line.replace("'", '"')
