import time

from ArgTest import ArgumentationFramework
# stable extensions
path = '/home/szczocik/Workspaces/Benchmark/B/'

# filenames                                                             # arguments | attacks
filename1 = '1/massachusetts_blockislandferry_2015-11-13.gml.80.tgf'    # 2         | 1
filename2 = '1/afinput_exp_cycles_indvary3_step4_batch_yyy03.tgf'       # 15        | 5
filename3 = '1/BA_40_80_4.tgf'                                          # 41        | 73
filename4 = '1/afinput_exp_cycles_indvary1_step8_batch_yyy08.tgf'       # 41        | 216
filename5 = '1/sembuster_60.tgf'                                        # 60        | 480


# full path
file = path + filename3
print('Reading file')
start = time.time()
af = ArgumentationFramework.read_tgf(file)
end = time.time()
print('File read in ' + str(end - start) + ' seconds')
print(len(af.frameworks))
af.draw_graph()
print('Number of arguments: ' + str(len(af.arguments)))
print('Number of attacks: ' + str(len(af.attacks)))
# af.draw_graph()
print('Creating Stable Extension')
start = time.time()
gr = af.get_stable_extension()
end = time.time()
print(gr)
print('Stable Extension created in ' + str(end - start) + ' seconds')

