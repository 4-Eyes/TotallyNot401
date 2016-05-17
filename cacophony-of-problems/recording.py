
def append_to_file(filename, string):
    file = open(filename, "a")
    file.write(string + "\n")
    file.close()