import ntpath
from collections import OrderedDict, defaultdict
from ArgTest.classes import *
import networkx as nx
import matplotlib.pyplot as plt
import numpy
from scipy import sparse
from ArgTest.classes import Attack
import itertools


class ArgumentationFramework(object):

    def __init__(self, name):
        self.name = name
        self.arguments = OrderedDict()
        self.attacks = []
        self.defenceSets = [set()]
        self.args_to_defence_sets = {}
        self.matrix = numpy.empty((0, 0))
        self.matrix_permutation = OrderedDict()

    def __str__(self):
        my_string = 'Argumentation Framework: \n'
        my_string += 'Argument\tAttacks\tAttacked by\n'
        for a in self.arguments:
            my_string += a
            my_string += '\t'
            my_string += '|'
            for att in self.arguments[a].attacking:
                my_string += att + ', '
            my_string += '\t|'
            for att in self.arguments[a].attacked_by:
                my_string += att + ', '
            my_string += '|\n'
        return my_string

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
        if attacker.name not in self.arguments:
            self.add_argument(attacker)
        if attacked.name not in self.arguments:
            self.add_argument(attacked)
        attack = Attack(attacker.name, attacked.name)
        self.attacks.append((attacker.name, attacked.name))
        self.arguments.get(attacker.name).add_attack(attacked.name)
        self.arguments.get(attacked.name).add_attacker(attacker.name)

    def get_dense_matrix(self):
        """
        :return: Dense Matrix
        """
        if type(self.matrix) is numpy.ndarray:
            self.matrix = self.create_matrix()
        return self.matrix.todense()

    def draw_graph(self):
        """
        Method to draw directed graph of the argumentation framework
        :return:
        """
        G = nx.DiGraph()

        for n in self.arguments.keys():
            G.add_node(n)
        for n in self.attacks:
            G.add_edge(n[0], n[1])
        pos = nx.spring_layout(G, k=0.30, iterations=20)
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos)
        plt.show()

    def get_attackers(self, arg_name):
        return self.arguments.get(arg_name).attacked_by

    def get_targets_of(self, arg_name):
        return self.arguments.get(arg_name).attacking

    def get_arguments_not_attacked(self):
        not_attacked = []
        for arg in self.arguments:
            if not self.arguments.get(arg).attacked_by:
                not_attacked.append(arg)
        return not_attacked

    def get_grounded_labelling(self):
        """
        Generate Grounded labelling
        :return: 
        """
        in_set = set(self.get_arguments_not_attacked())
        out_set = set()
        test = True
        grounded_ext = {
            'in': in_set.copy(),
            'out': set(),
            'undec': set()
        }

        while test:
            grounded_ext_copy = grounded_ext.copy()

            for arg in in_set:
                for next_arg in self.arguments.get(arg).attacking:
                    if set(self.arguments[next_arg].attacked_by).issubset(grounded_ext['in']):
                        out_set.add(next_arg)

            for arg in in_set:
                grounded_ext['in'].add(arg)
            in_set.clear()

            for arg in out_set:
                for next_arg in self.arguments.get(arg).attacking:
                    if set(self.arguments[next_arg].attacked_by).issubset(grounded_ext['out']):
                        in_set.add(next_arg)

            for arg in out_set:
                grounded_ext['out'].add(arg)
            out_set.clear()

            if grounded_ext == grounded_ext_copy:
                test = False

            for arg in self.arguments:
                if arg not in grounded_ext['in'] and arg not in grounded_ext['out']:
                    grounded_ext['undec'].add(arg)
        return grounded_ext

    def is_conflict_free(self, set_args):
        for arg in set_args:
            if set(self.arguments[arg].attacking) in set(set_args):
                return False
        return True

    def is_admissible(self, set_args):
        if self.is_conflict_free(set_args):
            attackers_of_set = self.get_attackers_of_set(set_args)
            if attackers_of_set:
                if attackers_of_set.issubset(self.get_attacks_of_set(set_args)):
                    return True
            else:
                return False
        return False

    def is_complete(self, set_args):
        if self.is_admissible(set_args):
            defended = self.get_defended_args_by(set_args)
            return len(set(set_args).intersection(defended)) == len(set(defended))
        else:
            return False

    def get_attackers_of_set(self, set_args):
        attackers = set()
        for arg in set_args:
            attackers.update(self.arguments[arg].attacked_by)
        return attackers

    def get_attacks_of_set(self, set_args):
        attacks = set()
        for arg in set_args:
            attacks.update(self.arguments[arg].attacking)
        return attacks

    def get_defended_args_by(self, set_args):
        defended = set()
        attacks_of_set = self.get_attacks_of_set(set_args)
        for arg in self.arguments:
            attackers = self.arguments[arg].attacked_by
            defen = True
            for att in attackers:
                if att not in attacks_of_set:
                    defen = False
                    break
            if defen:
                defended.add(arg)
        return defended

    def create_matrix(self):
        """
        Method used to create the matrix for the argumentation framework
        :return:
        """
        self.matrix = numpy.zeros((len(self.arguments), len(self.arguments)))
        for attack in self.attacks:
            self.matrix[self.arguments[attack[0]].mapping, self.arguments[attack[1]].mapping] = 1
        return sparse.coo_matrix(self.matrix)

    def get_subblocks_with_zeros(self):
        """
        Method to generate all sub blocks from original matrix where all values are 0's
        :return: list of sets of indexes for matrix where values in corresponding rows/columns are 0's
        """
        self.matrix = self.create_matrix()
        my_matrix = self.matrix.todense()
        my_return = []
        test = defaultdict(set)
        zeros = numpy.where(my_matrix == 0)
        for k, v in zip(zeros[0], zeros[1]):
            test[k].add(v)

        for v in range(0, my_matrix.shape[0]):
            combinations = itertools.combinations(range(my_matrix.shape[0]), v + 1)
            for comb in combinations:
                my_sets = [test[x] for x in comb]
                intersection = set(comb).intersection(*my_sets)
                if len(intersection) == len(comb):
                    my_return.append(intersection)
        return my_return

    def get_stable_extension(self):
        """
        Function to get all stable extension of the argumentation framework
        :return: list of sets of arguments that are stable extensions
        """
        my_stable_extension = []
        my_range = range(self.matrix.shape[0])
        s_subblock = self.get_subblocks_with_zeros()
        for v in s_subblock:
            columns = set(my_range).symmetric_difference(v)
            if type(self.matrix) is numpy.ndarray:
                self.matrix = self.create_matrix()
            rows = self.matrix.todense()[list(v), :]
            test_matrix = rows[:, list(columns)]
            zeros = numpy.where(test_matrix == 0)
            if len(zeros[0]) == 0:
                my_stable_extension.append(v)
        return my_stable_extension

    def is_stable_extension(self, arguments):
        """
        Verifies if the provided argument(s) are stable extension using matrix
        :param arguments: list of arguments to be checked
        :return: True if the provided arguments are a stable extension, otherwise False
        """
        my_labels = [self.arguments[x].mapping for x in arguments]
        my_submatrix = self.__get_submatrix(my_labels, my_labels)
        if len(numpy.where(my_submatrix == 1)[0]) > 0:
            return False
        my_column_vertices = self.__get_submatrix(my_labels, [x for x in
                                                              set(range(len(self.arguments))).symmetric_difference(
                                                                  my_labels)])
        if len(numpy.where(my_column_vertices == 0)[0]) > 0:
            return False
        return True

    def __get_submatrix(self, rows, columns):
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

    @staticmethod
    def read_tgf(path):
        try:
            from pyparsing import Word, alphanums, ZeroOrMore, White, Suppress, Group, ParseException, Optional
        except ImportError:
            raise ImportError("read_tgf requires pyparsing")

        if not isinstance(path, str):
            return

        # Define tgf grammar
        s = White(" ")
        tag = Word(alphanums)
        arg = Word(alphanums)
        att = Group(arg + Suppress(s) + arg + Optional(Suppress(s) + tag))
        nl = Suppress(White("\n"))

        graph = Group(ZeroOrMore(arg + nl)) + Suppress("#") + nl + Group(ZeroOrMore(att + nl) + ZeroOrMore(att))

        f = open(path, 'r')
        f = f.read()

        head, tail = ntpath.split(path)
        framework = ArgumentationFramework(tail)

        try:
            parsed = graph.parseString(f)
        except ParseException as e:
            raise e

        for arg in parsed[0]:
            framework.add_argument(arg)

        for att in parsed[1]:
            framework.add_attack(att[0], att[1])

        return framework
