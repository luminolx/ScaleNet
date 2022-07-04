import sys

log_path = sys.argv[1]

log = open(log_path, 'r')
nets = []
last = ''
for line in log:
    if ']' not in line:
        last += line[:-1]
        continue
    line = last + line
    last = ''
    print(line[:-1])
    sp = line.split('-')
    if len(sp) != 3:
        continue
    nets.append([' '.join([x for x in sp[0].split(' ') if x != '']), float(sp[1]), int(sp[2])])

nets.sort(key=lambda x: x[1], reverse=True)
print('===================result========================')
print('\n'.join([str(x) for x in nets]))
print('===================top 10========================')
print('\n'.join([str(x) for x in nets[:10]]))
print('==================generate=======================')
print('\n'.join(['s = \'' + x[0][2:-1] + '\'  # ' + '{} {}'.format(x[1], x[2]) for x in nets][:10]))
print('================FLOPs top 10=====================')
print('\n'.join([str(x) for x in nets if 318e6 < x[2] < 322e6][:10]))
print('==================generate=======================')
print('\n'.join(['s = \'' + x[0][1:-1] + '\'  # ' + '{} {}'.format(x[1], x[2]) for x in nets if 318e6 < x[2] < 322e6][:10]))


