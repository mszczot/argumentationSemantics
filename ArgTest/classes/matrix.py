from collections import defaultdict
import numpy
from scipy import sparse
from itertools import combinations


class Matrix(object):
    def __init__(self, arguments=[], attacks=[]):
        """
        Object constructor
        :param arguments: List of arguments objects
        :param attacks: List of attacks
        """
        self._matrix = self.__create_matrix(arguments, attacks)

    @property
    def shape(self):
        return self._matrix.shape

    @property
    def to_dense(self):
        return self._matrix.todense()

    @staticmethod
    def __create_matrix(arguments, attacks):
        """
        Method used to create the matrix for the argumentation framework
        :return:
        """
        return sparse.coo_matrix(([1] * len(attacks), ([arguments[v[0]].mapping for v in attacks],
                                                       [arguments[v[1]].mapping for v in attacks])),
                                 shape=(len(arguments), len(arguments)))

    def get_sub_matrix(self, rows, columns):
        """
        Gets submatrix from the main matrix based on the rows and columns provided
        :param rows: indexes of rows to be included
        :param columns: indexes of columns to be included
        :return: sub matrix of original matrix limited to rows and columns provided
        """
        rows = self.to_dense[list(rows), :]
        return rows[:, list(columns)]

    def get_sub_blocks_with_zeros(self):
        """
        Method to generate all sub blocks from original matrix where all values are 0's. Not efficient
        :return: list of sets of indexes for matrix where values in corresponding rows/columns are 0's
        """
        my_matrix = self.to_dense
        my_return = []
        test = defaultdict(set)
        zeros = numpy.where(my_matrix == 0)
        for k, v in zip(zeros[0], zeros[1]):
            test[k].add(v)
        for v in range(0, my_matrix.shape[0]):
            possible_combinations = combinations(range(my_matrix.shape[0]), v + 1)
            for comb in possible_combinations:
                my_sets = [test[x] for x in comb]
                intersection = set(comb).intersection(*my_sets)
                if len(intersection) == len(comb):
                    my_return.append(list(intersection))
        return my_return
