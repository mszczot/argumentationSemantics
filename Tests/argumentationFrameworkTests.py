from unittest import TestCase
from nose_parameterized import parameterized
from ArgTest import *
from Tests import TestHelper


class ArgumentationFrameworkTests(TestCase):
    prefix = './frameworks/'

    def setUp(self):
        """METHOD_SETUP"""

    def tearDown(self):
        """METHOD_TEARDOWN"""

    @parameterized.expand([
        [prefix + 'stable1.tgf', prefix + 'stable1answer'],
        [prefix + 'stable2.tgf', prefix + 'stable2answer'],
    ])
    def stable_extension_test(self, framework, solution):
        argumentation_framework = ArgumentationFramework.read_tgf(framework)
        actual_stable = argumentation_framework.get_stable_extension()
        expected_stable = TestHelper.read_solution_from_file(solution)
        self.assertEqual(expected_stable, actual_stable)
