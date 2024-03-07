from torch.utils.data import Dataset
import numpy as np
from datetime import datetime
from collections import Counter


class HPEMBData(Dataset):
    def __init__(self, file_path, neg_size, hist_len, hist_node, directed=False, transform=None):
        self.neg_size = neg_size
        self.transform = transform
        self.directed = directed
        self.hist_len = hist_len
        self.hist_node = hist_node
        self.NEG_SAMPLING_POWER = 0.75
        self.neg_table_size = int(1e6)
        self.node2hist = {}
        self.degree = {}
        self.timeinterval = 1
        self.path = file_path
        self.node, self.ts = self.process_whole_graph()
        self.ts = self.ts[0:50]
        self.node_dim = len(self.node)
        self.data_size = 0

        for i in range(len(self.ts)):
            self.node2hist[i] = {}
            self.degree[i] = dict()

        for i in range(len(self.ts)):
            for j in range(len(self.node)):
                self.node2hist[i][self.node[j]] = list()
                self.degree[i][self.node[j]] = list()

        for i in range(len(self.ts)):
            sub_path = './data/' + str(i) + '.txt'
            with open(sub_path, 'r') as infile:
                for line in infile:
                    parts = line.split()
                    s = int(float(parts[0]))
                    t = float(parts[1])
                    self.node2hist[i][s].append(t)
                    if not directed:
                        self.node2hist[i][t].append(s)

                for j in range(len(self.node)):
                    deg = len(self.node2hist[i][self.node[j]])
                    self.degree[i][self.node[j]].append(deg)
                    x = Counter(self.node2hist[i][self.node[j]])
                    self.data_size += len(x)
                    self.node2hist[i][self.node[j]] = {}
                    self.node2hist[i][self.node[j]] = dict(x)

        # self.neg_table = np.zeros((len(self.ts), self.neg_table_size), dtype=np.int32)
        # self.init_neg_table()

        self.idx2source_id = np.zeros((self.data_size,), dtype=np.int32)
        self.idx2target_id = np.zeros((self.data_size,), dtype=np.int32)
        self.idx2ts = np.zeros((self.data_size,), dtype=np.int32)
        idx = 0
        for ts_idx in range(len(self.ts)):
            for s_node in self.node2hist[ts_idx]:
                for t_node in self.node2hist[ts_idx][s_node]:
                    self.idx2source_id[idx] = s_node
                    self.idx2target_id[idx] = t_node
                    self.idx2ts[idx] = ts_idx
                    idx += 1

    def process_whole_graph(self):
        try:
            files = os.listdir(self.path)
            num_files = len(files)
            ts = [i for i in range(num_files - 1)]
            t0 = 1000000000
            with open(self.path + "raw_dataset.txt", 'r') as infile:
                tmp = []
                for line in infile:
                    parts = line.split()
                    t = int((int(parts[2]) - t0) / (24 * 3600 * 7))
                    parts[2] = t
                    parts[0] = int(parts[0])
                    parts[1] = int(parts[1])
                    tmp.append(parts)
            data = np.array(tmp)
            node = np.unique(data_new[:, [0, 1]])
            return node, ts
        except FileNotFoundError:
            return "Directory not found"

    def init_neg_table(self):
        for i in range(len(self.ts)):
            tot_sum, cur_sum, por = 0., 0., 0.
            n_id = 0
            for k in range(self.node_dim):
                tot_sum += np.power(self.degree[i][self.node[k]], self.NEG_SAMPLING_POWER)
            for k in range(self.neg_table_size):
                if (k + 1.) / self.neg_table_size > por:
                    cur_sum += np.power(self.degree[i][self.node[n_id]], self.NEG_SAMPLING_POWER)
                    por = cur_sum / tot_sum
                    n_id += 1
                self.neg_table[i, k] = self.node[n_id-1]

    def get_node_dim(self):
        return self.node_dim

    def get_ts_length(self):
        return len(self.ts)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        s_node = self.idx2source_id[idx]
        t_node = self.idx2target_id[idx]
        t_time = self.idx2ts[idx]
        t_weight = self.node2hist[t_time][s_node][t_node]

        neg_nodes = self.negative_sampling_new(t_time)

        hist_node = np.zeros((self.hist_len, self.hist_node))
        hist_weight = np.zeros((self.hist_len, self.hist_node))
        hist_mask = np.zeros((self.hist_len, self.hist_node))
        hist_t = np.zeros(self.hist_len)

        if t_time - self.hist_len < 0:
            for i in range(t_time):
                keys = np.fromiter(self.node2hist[i][s_node].keys(), dtype=float)
                value = np.fromiter(self.node2hist[i][s_node].values(), dtype=float)
                hist_t[i] = t_time - self.hist_len + i
                if np.sum(value) != 0:
                    prob = value / np.sum(value)
                    size_value = np.minimum(self.hist_node, len(prob))
                    hist_node_tmp = np.random.choice(keys, size=size_value, replace=False, p=prob)
                    hist_node[i][:len(hist_node_tmp)] = hist_node_tmp
                    hist_mask[i][:len(hist_node_tmp)] = 1.
        else:
            for i in range(self.hist_len):
                keys = np.fromiter(self.node2hist[t_time - self.hist_len + i][s_node].keys(), dtype=float)
                value = np.fromiter(self.node2hist[t_time - self.hist_len + i][s_node].values(), dtype=float)
                hist_t[i] = t_time - self.hist_len + i
                if np.sum(value) != 0:
                    prob = value / np.sum(value)
                    size_value = np.minimum(self.hist_node, len(prob))
                    hist_node_tmp = np.random.choice(keys, size=size_value, replace=False, p=prob)
                    hist_node[i][:len(hist_node_tmp)] = hist_node_tmp
                    hist_mask[i][:len(hist_node_tmp)] = 1.
        hist_t[hist_t < 0] = 0

        sample = {
            'source_node': s_node,
            'target_node': t_node,
            'target_time': t_time,
            'target_weight': t_weight,
            'neg_nodes': neg_nodes,
            'hist_node': hist_node,
            'hist_mask': hist_mask,
            'hist_time': hist_t,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def negative_sampling(self, time):
        rand_idx = np.random.randint(0, self.neg_table_size, (self.neg_size,))
        sampled_nodes = self.neg_table[time][rand_idx]
        return sampled_nodes

    def negative_sampling_new(self, time):
        rand_idx = np.random.randint(0, self.node_dim, (self.neg_size,))
        return rand_idx