from collections import OrderedDict
from ArgTest.classes import *
import networkx as nx
import matplotlib.pyplot as plt


class ArgumentationFramework(object):

    def __init__(self, name):
        self.name = name
        self.arguments = OrderedDict()
        self.attacks = []

    def __str__(self):
        str = 'Argumentation Framework: \n'
        for a in self.arguments:
            str += a
            str += ', '
        return str

    def add_argument(self, argument):
        if argument not in self.arguments:
            self.arguments[argument] = Argument(argument)

    def add_attack(self, attacker, attacked):
        attacker = self.arguments.get(attacker)
        attacked = self.arguments.get(attacked)
        if attacker.name not in self.arguments:
            self.add_argument(attacker)
        if attacked.name not in self.arguments:
            self.add_argument(attacked)
        self.attacks.append((attacker.name, attacked.name))
        self.arguments.get(attacker.name).add_attack(attacked.name)
        self.arguments.get(attacked.name).add_attacker(attacker.name)

    def draw_graph(self):
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