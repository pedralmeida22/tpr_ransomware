import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputFile', nargs='?', required=True, help='input file')
    parser.add_argument('-o', '--outputFile', nargs='?', required=True, help='format', default=1)

    args = parser.parse_args()

    f = open(args.inputFile, "r")
    linhas = f.read()
    array = linhas.split("\n")
    print(len(array))

    fW = open(args.outputFile, "w")

    for linha in array:
        splite = linha.split(" ")
        if len(splite) > 1:
            fW.write(str(splite[2]) + " " + str(splite[4]) + "\n")


if __name__ == '__main__':
    main()
