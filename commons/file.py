# python imports
import os


def fmt_list(value, delimiter=','):
    return delimiter.join(str(i) for i in value)


def fmt_matrix(cm):
    tmp = list()
    for row in cm.tolist():
        for col in row:
            tmp.append(col)
    return tmp


def calculate_mean_matrix(matrix):
    mean_matrix = list()
    for i in range(len(matrix[0])):
        sum_column = 0
        for row in matrix:
            sum_column += row[i]
        mean_matrix.append(sum_column / len(matrix))
    return mean_matrix


def read_file(filename, cast=float, delimiter=','):
    tmp = list()
    with open(filename) as f:
        for line in f:
            tmp.append([cast(i) for i in line.rstrip().split(delimiter)])
    return tmp


def to_file(filename, text):
    # check and create folder it doesn't exists
    dirs = os.path.split(filename)[0]
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    with open(filename, 'a') as f:
        f.write(text)


def metrics_to_file(base_url, metrics):
    names = ['acc', 'time', 'precision', 'recall', 'f1', 'cm']

    for metric, name in zip(metrics, names):
        values = fmt_list(metric, "-") if name == 'cm' else fmt_list(metric)
        to_file(f'{base_url}-{name}.txt', f'{values}\n')
