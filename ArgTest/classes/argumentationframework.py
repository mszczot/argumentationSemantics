import ntpath
from collections import OrderedDict, defaultdict
# from ArgTest.classes import Argument, Attack, Framework
import networkx as nx
import matplotlib.pyplot as plt
import numpy
from scipy import sparse
import itertools
from ArgTest.classes.framework import Framework
from ArgTest.classes.database import Database


class ArgumentationFramework(object):

    def __init__(self, name):
        self.name = name
        self.frameworks = {}
        self.arguments = OrderedDict()
        self.attacks = []
        self.attacks_mappings = []
        self.defenceSets = [set()]
        self.args_to_defence_sets = {}
        self.matrix = numpy.empty((0, 0))
        self.matrix_permutation = OrderedDict()
        self.zero_blocks = set()
        self.database = Database()

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

    def add_argument(self, arg):
        """
        Method to add argument to argumentation framework
        :param argument: Name of the argument to be added
        :return:
        """
        # if argument not in self.arguments:
        #     self.arguments[argument] = Argument(argument, len(self.arguments))
        #     self.__add_element_to_zero_blocks(argument)
        if len(self.frameworks) == 0:
            counter = len(self.frameworks)
            self.frameworks[counter] = Framework(counter)
            self.frameworks[counter].add_argument(arg)
            self.arguments[arg] = counter
        else:
            if arg not in self.arguments:
                counter = len(self.frameworks)
                self.frameworks[counter] = Framework(counter)
                self.frameworks[counter].add_argument(arg)
                self.arguments[arg] = counter

    def get_argument_from_mapping(self, mapping):
        for v in self.arguments:
            if self.arguments[v].mapping == mapping:
                return self.arguments[v].name
        return None

    def __add_element_to_zero_blocks(self, arg_name):
        arg_name = self.arguments[arg_name].mapping
        zero_blocks = self.zero_blocks.copy()
        if len(self.zero_blocks) == 0:
            # self.zero_blocks.add(frozenset([arg_name]))
            # self.database.add_conflict_free_set(arg_name)
            return
        else:
            for v in zero_blocks:
                new_set = set(v)
                new_set.add(arg_name)
                # self.zero_blocks.add(frozenset(new_set))
                # self.database.add_conflict_free_set(new_set)
            # self.zero_blocks.add(frozenset([arg_name]))
            # self.database.add_conflict_free_set(arg_name)

    def __remove_attack_from_zero_blocks(self, att):
        to_be_removed = set(frozenset(x) for x in self.zero_blocks if set(att).issubset(x))
        self.zero_blocks = self.zero_blocks - to_be_removed

    def get_keys(self, value):
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
        attacked_counter = self.frameworks[self.arguments[attacked]].counter
        attacker_counter = self.frameworks[self.arguments[attacker]].counter
        if attacker_counter != attacked_counter:
            self.frameworks[self.arguments[attacker]].merge_framework_through_attack(self.frameworks[self.arguments[attacked]], attacker, attacked)
            del self.frameworks[attacked_counter]
            self.arguments[attacked] = attacker_counter
            for k in self.get_keys(attacked_counter):
                self.arguments[k] = attacker_counter
        else:
            self.frameworks[self.arguments[attacker]].add_attack(attacker, attacked)
        self.attacks.append((attacker, attacked))
        # if attacker.name not in self.arguments:
        #     self.add_argument(attacker)
        # if attacked.name not in self.arguments:
        #     self.add_argument(attacked)
        # attack = Attack(attacker.name, attacked.name)
        # self.attacks.append((attacker.name, attacked.name))
        # self.arguments.get(attacker.name).add_attack(attacked.name)
        # self.arguments.get(attacked.name).add_attacker(attacker.name)
        # # self.__remove_attack_from_zero_blocks(set([attacker.mapping, attacked.mapping]))
        # self.attacks_mappings.append(tuple([attacker.mapping, attacked.mapping]))

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
        # s_subblock = self.get_subblocks_with_zeros()
        s_subblock = self.get_conflict_free_from_matrix()
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

    def get_stable_extension_test_1(self):
        """
        Method to test if can get stable extensions only from the list of rows/columns in matrix where value is 0
        :return:
        """
        # TODO Should remove all elements which are not attacked nor attacking any other element first
        my_return = set()
        for f in self.frameworks:
            framework = self.frameworks[f]
            framework.matrix = framework.create_matrix()
            zeros = numpy.where(framework.matrix.todense() == 0)
            sets_to_check = defaultdict(list)
            for k, v in zip(zeros[0], zeros[1]):
                sets_to_check[k].append(framework.get_argument_from_mapping(v))
            for v in sets_to_check:
                if self.is_stable_extension(framework, sets_to_check[v]):
                    if set(sets_to_check[v]) not in set(my_return):
                        # my_return[framework.counter].append(sets_to_check[v])
                        for element in sets_to_check[v]:
                            my_return.add(element)
        return my_return

    def is_stable_extension(self, framework, args):
        """
        Verifies if the provided argument(s) are stable extension using matrix
        :param args: list of arguments to be checked
        :return: True if the provided arguments are a stable extension, otherwise False
        """
        my_labels = [framework.arguments[x].mapping for x in args]
        # get the arguments that are not attacking nor are attacked - they will be part of the stable extension,
        # but won't be checked by vertices
        if type(framework.matrix) is numpy.ndarray:
            framework.matrix = framework.create_matrix()
        x = numpy.where(framework.matrix.todense() == 1)
        y = defaultdict(list)
        for k, v in zip(x[0], x[1]):
            y[k].append(v)
        # TODO this throws error when there are no elements which are not attacking nor are attacked
        not_attacked_or_attacking = None #[x for x in set(my_labels).symmetric_difference(y.keys()) if x not in y.values()]
        my_submatrix = framework.get_submatrix(my_labels, my_labels)
        if len(numpy.where(my_submatrix == 1)[0]) > 0:
            return False
        a = [x for x in set(range(len(framework.arguments))).symmetric_difference(my_labels)]
        # my_column_vertices = self.__get_submatrix(set(my_labels) - set(not_attacked_or_attacking), [x for x in
        #                                                       set(range(len(self.arguments))).symmetric_difference(
        #                                                           my_labels)])
        my_column_vertices = framework.get_submatrix(set(my_labels), [x for x in
                                                              set(range(len(framework.arguments))).symmetric_difference(
                                                                  my_labels)])
        if not_attacked_or_attacking is None:
            if len(numpy.where(my_column_vertices == 0)[0]) > 0:
                return False
        else:
            if len(set(numpy.where(my_column_vertices == 1)[0])) == my_column_vertices.shape[0]:
                return True
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

    """ Just for testing """
    def test1(self):
        self.matrix = self.create_matrix()
        test = defaultdict(list)
        ones = numpy.where(self.matrix.todense() == 1)
        for k, v in zip(ones[0], ones[1]):
            test[k].append(v)
        print(test)

    """ Just for testing """
    def test(self):
        test = defaultdict(list)
        test1 = defaultdict(list)
        zeros = numpy.where(self.matrix.todense() == 0)
        ones = numpy.where(self.matrix.todense() == 1)
        for k, v in zip(zeros[0], zeros[1]):
            test[k].append(v)
        for k, v in zip(ones[0], ones[1]):
            test1[k].append(v)
        my_return = defaultdict(list)
        def common_entries(*dcts):
            for i in set(dcts[0]).intersection(*dcts[1:]):
                yield (i,) + tuple(d[i] for d in dcts)
        for v in common_entries(test, test1):
            my_return[v[0]].append(v[1])
            my_return[v[0]].append(v[2])
        print(my_return)

    def get_conflict_free_from_matrix(self):
        """
        Generator to get all conflict free sets from matrix using the principle of 0's sub blocks
        Replacing method get_subblocks_with_zeros
        :return:
        """
        for v in range(len(self.arguments)):
            v += 1
            combinations = itertools.combinations(range(len(self.arguments)), v)
            for comb in combinations:
                if len(comb) > 1:
                    test = True
                    for att in self.attacks_mappings:
                        if set(att).issubset(set(comb)):
                            test = False
                            break
                    if test:
                        yield [x for x in comb]
                else:
                    yield [x for x in comb]

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
