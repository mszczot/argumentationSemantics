from unittest import TestCase
from ArgTest import *


class ArgumentationFrameworkTests(TestCase):

    def stable_extension_test(self):
        argumentation_framework = ArgumentationFramework.read_tgf('./frameworks/stable1.tfg')
        actual_stable = argumentation_framework.get_stable_extension()
        file = open('./frameworks/stable1answer', 'r')
        expected_stable = file.read()
        self.assertEqual(expected_stable, actual_stable)
