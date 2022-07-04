import matplotlib.pyplot as plt

f = open('statistic_flops.txt', 'r')
data = []
for line in f:
    if len(line) == 0:
        continue
    flops = int(line.split(',')[-1])
    data.append(flops / 1e6)

bins = [x*50 for x in range(1, 16)]
plt.hist(data, bins=bins, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
plt.xlabel('FLOPs')
plt.ylabel('number')
plt.title('FLOPs statistics')
plt.savefig('flops_statistics.png')
