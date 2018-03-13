from scipy import sparse

import numpy

from ArgTest.classes import Argument, Attack


class Framework(object):
    def __init__(self, counter):
        self.arguments = {}
        self.attacks = []
        self.counter = counter
        self.matrix = numpy.ndarray

    def add_argument(self, argument):
        """
        Method to add argument to argumentation framework
        :param argument: Name of the argument to be added
        :return:
        """
        if argument not in self.arguments:
            self.arguments[argument] = Argument(argument, len(self.arguments))

    def add_attack(self, attacker, attacked):
        """
        Method to add attack to argumentation framework
        :param attacker: argument that attacks 'attacked'
        :param attacked: argument attacked by 'attacker'
        :return:
        """
        attacker = self.arguments.get(attacker)
        attacked = self.arguments.get(attacked)
        if attacker not in self.arguments:
            self.add_argument(attacker)
        if attacked not in self.arguments:
            self.add_argument(attacked)
        attack = Attack(attacker, attacked)
        self.attacks.append((attacker, attacked))
        self.arguments.get(attacker).add_attack(attacked)
        self.arguments.get(attacked).add_attacker(attacker)

    def merge_framework(self, framework):
        self.arguments = {**self.arguments, **framework.arguments}
        self.attacks = self.attacks + framework.attacks

    def merge_framework_through_attack(self, framework, attacker, attacked):
        self.arguments = {**self.arguments, **framework.arguments}
        counter = 0
        for a in self.arguments.values():
            a.mapping = counter
            counter += 1
        self.attacks = self.attacks + framework.attacks
        self.attacks.append(tuple([attacker, attacked]))

    def create_matrix(self):
        """
        Method used to create the matrix for the argumentation framework
        :return:
        """
        self.matrix = numpy.zeros((len(self.arguments), len(self.arguments)))
        for attack in self.attacks:
            self.matrix[self.arguments[attack[0]].mapping, self.arguments[attack[1]].mapping] = 1
        return sparse.coo_matrix(self.matrix)

    def get_argument_from_mapping(self, mapping):
        for v in self.arguments:
            if self.arguments[v].mapping == mapping:
                return self.arguments[v].name
        return None

    def get_submatrix(self, rows, columns):
        """
        Gets submatrix from the main matrix based on the rows and columns provided
        :param rows: indexes of rows to be included
        :param columns: indexes of columns to be included
        :return: sub matrix of original matrix limited to rows and columns provided
        """
        if type(self.matrix) is numpy.ndarray:
            self.matrix = self.create_matrix()
        rows = self.matrix.todense()[list(rows), :]
        return rows[:, list(columns)]
