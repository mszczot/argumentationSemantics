# test from the paper:
# arguments: a, b, c, d
# attacks: (a,b), (b,a), (b,c), (c,d), (d,c)
# complete extension: empty set, {a}, {d}, {a,c}, {a,d}, {b,d}
import time

from ArgTest import ArgumentationFramework

af = ArgumentationFramework('complete')
af.add_argument('a')
af.add_argument('b')
af.add_argument('c')
af.add_argument('d')
af.add_attack('a', 'b')
af.add_attack('b', 'a')
af.add_attack('b', 'c')
af.add_attack('c', 'b')
af.add_attack('d', 'b')

print(af.get_dense_matrix())
print(af.get_stable_extension())
print(af.is_stable_extension(['a', 'b', 'c']))
print(af.is_stable_extension(['a', 'c', 'd']))