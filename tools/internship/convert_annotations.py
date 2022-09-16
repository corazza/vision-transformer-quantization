import IPython
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

M = open(input_file).read().strip().split("\n")
print(M[:3])
M = [r.split("\t")[:2] for r in M]
print(M[:3])
with open(output_file, 'w') as f: 
    for line in M:
        f.write(f'images/{line[0]} {line[1]}')
        f.write('\n')
        