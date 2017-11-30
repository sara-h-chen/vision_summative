with open("logs.txt", "r") as infile, open("logs_2.csv", "w") as outfile:
    for line in infile:
        new_string = line.replace("(","")
        new_string = new_string.replace(")", "\n")
        outfile.write(new_string)
