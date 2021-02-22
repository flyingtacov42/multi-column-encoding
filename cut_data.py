import csv

length = 100000
output = []
filename = "dmv.csv"
with open(filename, "r", newline="", encoding="ISO8859") as csvfile:
    reader = csv.reader(csvfile)
    count = 0
    for row in reader:
        output.append(row)
        count += 1
        if count >= length:
            break

filename[:-4] + str(length) + ".csv"
with open("dmv100000.csv", "w", newline="", encoding="ISO8859") as csvfile:
    writer = csv.writer(csvfile)
    # writer.writerow(["Date", "Time", "Location", "Code"])
    for row in output:
        writer.writerow(row)

