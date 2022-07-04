import math
res = {}
f = open('statistic_flops.txt', 'r')
for line in f:
    if len(line) == 0:
        continue
    flops = int(line.split(',')[-1])
    f_quanti = math.floor(flops / 1e8) 
    if res.get(f_quanti) is None:
        res[f_quanti] = 0
    res[f_quanti] += 1

print(res)

res_l = []
for k in res:
    res_l.append((k, res[k]))

res_l.sort(key=lambda x: x[0])

total = sum([x[1] for x in res_l])
print('total num: {}'.format(sum([x[1] for x in res_l])))
print(', '.join(['{}00M: {}'.format(x[0], x[1]) for x in res_l]))
print(', '.join(['{}00M: {:.3f}'.format(x[0], x[1] / total) for x in res_l]))

