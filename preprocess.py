from torch.utils.data import Dataset
import numpy as np
from datetime import datetime
from collections import Counter

def process_whole_graph(path):
    t0 = 1000000000
    with open(path, 'r') as f:
        tmp = []
        for line in f:
            parts = line.split()
            t = int((int(parts[2]) - t0) / (24 * 3600 * 7))
            parts[2] = t
            parts[0] = int(parts[0])
            parts[1] = int(parts[1])
            tmp.append(parts)
    data = np.array(tmp)
    day = np.unique(data[:, 2])
    for i in range(len(day)):
        idx = np.where(data[:, 2] == day[i])[0]
        sub_ds = data[idx]
        path = './data/%d.txt' % (i)
        np.savetxt(path, sub_ds, fmt="%s")
    return

if __name__ == "__main__":
    preprocess = process_whole_graph('./whole-graph.txt')
