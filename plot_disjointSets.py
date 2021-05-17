import matplotlib.pyplot as plt

name = "stock"
with open("{}_plots/{}_encoding.txt".format(name, name), "r") as fin:
    rows = fin.readlines()
    rows = [row.split() for row in rows]
    fprs = [float(row[-1]) for row in rows]
    plt.hist(fprs, bins=len(rows) // 2)
    plt.title("{} FPRs".format(name))
    plt.xlabel("False positive rate improvement")
    plt.ylabel("Frequency")
    plt.savefig("{}_fpr.png".format(name))