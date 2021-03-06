import ntpath
from collections import OrderedDict, defaultdict
from scipy import sparse

import networkx as nx
import matplotlib.pyplot as plt
import numpy
from ArgTest.classes import Matrix, Argument, Framework


class ArgumentationFramework(object):

    def __init__(self, name):
        self.name = name  # Name of the framework
        self.frameworks = {}  # Dictionary of the sub frameworks
        self.arguments = OrderedDict()  # collection of all arguments in the framework
        self.attacks = []  # collection of all attacks in the framework
        self._matrix = None  # matrix representation of the framework

    @property
    def matrix(self):
        if self._matrix is None:
            self._matrix = Matrix(self.arguments, self.attacks)
        return self._matrix

    def __str__(self):
        """
        String representation of the Argumentation Framework
        :return: 
        """
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

    def add_argument(self, arg):
        """
        Method to add argument to argumentation framework
        :param arg: Name of the argument to be added
        :return:
        """
        if arg not in self.arguments:
            counter = len(self.arguments)
            self.arguments[arg] = Argument(arg, counter)

    def get_argument_from_mapping(self, mapping):
        """
        Method to get argument name from the mapping in the matrix
        :param mapping: row/column index in the matrix representation of the AF
        :return: 
        """
        for v in self.arguments:
            if self.arguments[v].mapping == mapping:
                return self.arguments[v].name
        return None

    def __get_keys(self, value):
        """
        Method to get all keys from arguments dict which have value 'value'
        :param value: value to find
        :return: 
        """
        my_return = []
        for k, v in self.arguments.items():
            if v == value:
                my_return.append(k)
        return my_return

    def add_attack(self, attacker, attacked):
        """
        Method to add attack to argumentation framework
        :param attacker: argument that attacks 'attacked'
        :param attacked: argument attacked by 'attacker'
        :return:
        """
        if attacker not in self.arguments:
            self.add_argument(attacker)
        if attacked not in self.arguments:
            self.add_argument(attacked)
        self.attacks.append((attacker, attacked))
        self.arguments[attacker].attacking.append(attacked)
        self.arguments[attacked].attacked_by.append(attacker)

    def draw_graph(self):
        """
        Method to draw directed graph of the argumentation framework
        :return:
        """
        graph = nx.DiGraph()

        for n in self.arguments.keys():
            graph.add_node(n)
        for n in self.attacks:
            graph.add_edge(n[0], n[1])
        pos = nx.spring_layout(graph, k=0.30, iterations=20)
        nx.draw_networkx_nodes(graph, pos)
        nx.draw_networkx_labels(graph, pos)
        nx.draw_networkx_edges(graph, pos)
        plt.show()

    def get_arguments_not_attacked(self):
        # TODO This won't work in the current setup - dict of arguments does not hold argument objects
        not_attacked = []
        for arg in self.arguments:
            if not self.arguments.get(arg).attacked_by:
                not_attacked.append(arg)
        return not_attacked

    def get_grounded_labelling(self):
        """
        TODO Needs to be reviewed in terms of the matrix
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

    def get_stable_extension(self):
        """
        Method to test if can get stable extensions only from the list of rows/columns in matrix where value is 0
        :return:
        """
        # TODO Should remove all elements which are not attacked nor attacking any other element first
        my_return = []
        zeros = numpy.where(self.matrix.to_dense == 0)
        sets_to_check = defaultdict(list)
        for k, v in zip(zeros[0], zeros[1]):
            sets_to_check[k].append(self.get_argument_from_mapping(v))
        for v in sets_to_check:
            to_test = self.get_conflict_free_from_set(sets_to_check[v])
            if self.__is_stable_extension(to_test):
                if set(to_test) not in set(my_return):
                    # my_return[framework.counter].append(sets_to_check[v])
                    my_return.append(frozenset(to_test))
        return my_return

    def __is_stable_extension(self, args):
        """
        Verifies if the provided argument(s) are stable extension using matrix
        :param framework: subframework of the Argumentation framework
        :param args: list of arguments to be checked
        :return: True if the provided arguments are a stable extension, otherwise False
        """
        # This is commented out as using zero sub blocks from matrix
        my_labels = [self.arguments[x].mapping for x in args]
        x = numpy.where(self.matrix.to_dense == 1)
        y = defaultdict(list)
        for k, v in zip(x[0], x[1]):
            y[k].append(v)
        # TODO this throws error when there are no elements which are not attacking nor are attacked
        not_attacked_or_attacking = None
        my_submatrix = self.matrix.get_sub_matrix(my_labels, my_labels)
        if len(numpy.where(my_submatrix == 1)[0]) > 0:
            return False
        # my_column_vertices = self.__get_submatrix(set(my_labels) - set(not_attacked_or_attacking), [x for x in
        #                                                       set(range(len(self.arguments))).symmetric_difference(
        #                                                           my_labels)])
        my_column_vertices = self.matrix.get_sub_matrix(set(my_labels), [x for x in
                                                                      set(range(len(
                                                                          self.arguments))).symmetric_difference(
                                                                          my_labels)])
        if not_attacked_or_attacking is None:
            for row in my_column_vertices.sum(axis=0).tolist():
                for v in row:
                    if v == 0:
                        return False
            # if len(numpy.where(my_column_vertices == 0)[0]) > 0:
            #     return False
        else:
            if len(set(numpy.where(my_column_vertices == 1)[0])) == my_column_vertices.shape[0]:
                return True
        return True

    def get_conflict_free_from_set(self, arg_set):
        my_labels = [self.arguments[x].mapping for x in arg_set]
        for arg in arg_set:
            arg_set = set(arg_set) - set(self.arguments[arg].attacking)
        return list(arg_set)

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
