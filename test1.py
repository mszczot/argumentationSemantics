from ArgTest import ArgumentationFramework


af = ArgumentationFramework('test')
af.add_argument('a11')
af.add_argument('a10')
af.add_argument('a13')
af.add_argument('a12')
af.add_argument('a15')
af.add_argument('a14')
af.add_argument('a17')
af.add_argument('a16')
af.add_argument('a19')
af.add_argument('a18')
af.add_argument('a1')
af.add_argument('a2')
af.add_argument('a3')
af.add_argument('a4')
af.add_argument('a5')
af.add_argument('a6')
af.add_argument('a7')
af.add_argument('a8')
af.add_argument('a9')

af.add_attack('a8','a10')
af.add_attack('a11','a12')
af.add_attack('a13','a4')
af.add_attack('a12','a11')
af.add_attack('a1','a7')
af.add_attack('a14','a2')
af.add_attack('a8','a18')
af.add_attack('a12','a5')
af.add_attack('a19','a13')
af.add_attack('a9','a15')
af.add_attack('a9','a14')
af.add_attack('a15','a9')
af.add_attack('a15','a6')
af.add_attack('a2','a10')
af.add_attack('a1','a17')
af.add_attack('a16','a4')
af.add_attack('a2','a14')
af.add_attack('a12','a18')
af.add_attack('a14','a9')
af.add_attack('a6','a3')
af.add_attack('a18','a5')
af.add_attack('a3','a19')
af.add_attack('a4','a13')
af.add_attack('a7','a1')
af.add_attack('a4','a16')
af.add_attack('a17','a1')
af.add_attack('a5','a11')
af.add_attack('a5','a12')
af.add_attack('a11','a5')
af.add_attack('a18','a12')
af.add_attack('a5','a18')
af.add_attack('a6','a15')
af.add_attack('a16','a17')
af.add_attack('a19','a3')
af.add_attack('a10','a2')
af.add_attack('a17','a16')
af.add_attack('a10','a8')


print(af.get_stable_extension())