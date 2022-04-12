# third-party imports
import numpy as np

# projects imports
from commons.file import read_file
from commons.file import to_file
from commons.file import fmt_list
from commons.file import calculate_mean_matrix
from commons.confidence import interval

# local imports
from constants import TECHNIQUES
from constants import LEARNERS
from constants import INPUT_RAW_FILE_PATH
from constants import OUTPUT_FILE_PATH


def process():

    for tech in TECHNIQUES:
        for learner in LEARNERS:

            # calculating mean, lower and upper values for
            # acc, precision, recall, f1 and learning time

            input = f'{INPUT_RAW_FILE_PATH}/{tech}'

            accs = read_file(f'{input}/{learner}-acc.txt')
            times = read_file(f'{input}/{learner}-time.txt')

            precisions = read_file(f'{input}/{learner}-precision.txt')
            recalls = read_file(f'{input}/{learner}-recall.txt')
            f1s = read_file(f'{input}/{learner}-f1.txt')

            results = [
                (accs, 'acc'),
                (times, 'time'),
                (precisions, 'precision'),
                (recalls, 'recall'),
                (f1s, 'f1')
            ]

            for values, name in results:
                mean_list = list()
                lower_list = list()
                upper_list = list()

                for col_index in range(len(values[0])):
                    col_elems = [row[col_index] for row in values]

                    mean = np.mean(col_elems)
                    lower, upper = interval(col_elems)

                    mean_list.append(mean)
                    lower_list.append(lower)
                    upper_list.append(upper)

                mean = fmt_list(mean_list)
                lower = fmt_list(lower_list)
                upper = fmt_list(upper_list)

                to_file(f'{OUTPUT_FILE_PATH}/{tech}-{name}-mean.txt', f'{mean}\n')
                to_file(f'{OUTPUT_FILE_PATH}/{tech}-{name}-lower.txt', f'{lower}\n')
                to_file(f'{OUTPUT_FILE_PATH}/{tech}-{name}-upper.txt', f'{upper}\n')

            # calculating mean, lower
            # and upper confusion matrices

            cms = read_file(f'{input}/{learner}-cm.txt', cast=str, delimiter='-')

            mean_matrices_list = list()
            for col_index in range(len(cms[0])):
                col_elems = [row[col_index] for row in cms]

                matrices_in_column = list()
                for row in col_elems:
                    matrices_in_column.append([float(i) for i in row.split(',')])
                mean_matrices_list.append(calculate_mean_matrix(matrices_in_column))

            mean_matrix = calculate_mean_matrix(mean_matrices_list)

            mean = fmt_list(mean_matrix)
            lower = fmt_list(mean_matrices_list[0])
            upper = fmt_list(mean_matrices_list[-1])

            to_file(f'{OUTPUT_FILE_PATH}/{tech}-cm-mean.txt', f'{mean}\n')
            to_file(f'{OUTPUT_FILE_PATH}/{tech}-cm-lower.txt', f'{lower}\n')
            to_file(f'{OUTPUT_FILE_PATH}/{tech}-cm-upper.txt', f'{upper}\n')
