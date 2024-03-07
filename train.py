import torch
from torch.autograd import Variable
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import numpy as np
import sys
from data_preprocess_facebook import HPEMBData
import os

FType = torch.FloatTensor
LType = torch.LongTensor
DID = 0


class HpEmb:
    def __init__(self, file_path, emb_size=64, neg_size=3, hist_len=3, hist_node=5, directed=False,
                 learning_rate=0.3, batch_size=10000, save_step=100, epoch_num=300):
        self.emb_size = emb_size
        self.neg_size = neg_size
        self.hist_len = hist_len
        self.hist_node = hist_node
        self.lr = learning_rate
        self.batch = batch_size
        self.save_step = save_step
        self.epochs = epoch_num
        self.para_prox = 0.1
        self.para_smth = 0.01
        self.data = HPEMBData(file_path, neg_size, hist_len, hist_node, directed)
        self.node_dim = self.data.get_node_dim()
        self.ts_length = self.data.get_ts_length()
        self.attn = 1
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                self.node_emb = Variable(torch.from_numpy(np.random.uniform(
                    -1. / np.sqrt(self.node_dim), 1. / np.sqrt(self.node_dim), (self.ts_length, self.node_dim, emb_size))).type(
                    FType).cuda(), requires_grad=True)

                self.delta = Variable((torch.zeros(self.node_dim) + 1.).type(FType).cuda(), requires_grad=True)
                self.W = torch.nn.Parameter(torch.zeros(size=(2*self.emb_size, 2*self.emb_size))).cuda()
                torch.nn.init.xavier_uniform_(self.W.data, gain=1.414)
                self.a = torch.nn.Parameter(torch.zeros(size=(2 * self.emb_size, 1))).cuda()
                torch.nn.init.xavier_uniform_(self.a.data, gain=1.414)
                self.leakyrelu = torch.nn.LeakyReLU(0.2)
        else:
            self.node_emb = Variable(torch.from_numpy(np.random.uniform(
                -1. / np.sqrt(self.node_dim), 1. / np.sqrt(self.node_dim), (self.ts_length, self.node_dim, emb_size))).type(
                FType), requires_grad=True)

            self.delta = Variable((torch.zeros(self.node_dim) + 1.).type(FType), requires_grad=True)
        self.opt = SGD(lr=learning_rate, params=[self.node_emb, self.delta])
        self.loss = torch.FloatTensor()

    def forward(self, s_node, t_node, t_time, t_weight, n_node, h_node, h_mask, hist_t):
        batch = s_node.size()[0]

        # For smooth loss
        smooth = (self.node_emb[1:] - self.node_emb[:-1]).to(self.device)
        smooth_new = (smooth ** 2).sum(dim=2).sum(dim=1)
        lsmooth = torch.mean(smooth_new)

        # For static loss
        s_node_emb = self.node_emb[t_time.type(LType), s_node.squeeze()].to(self.device)
        t_node_emb = self.node_emb[t_time.type(LType), t_node.squeeze()].to(self.device)
        time_expand_s = t_time.view(batch, -1).expand(batch, self.neg_size)
        n_node_emb = self.node_emb[time_expand_s.type(LType), n_node.squeeze()].to(self.device)

        s_node_emb_pre = self.node_emb[(t_time-1).type(LType), s_node.squeeze()].to(self.device)
        t_node_emb_pre = self.node_emb[(t_time-1).type(LType), t_node.squeeze()].to(self.device)
        n_node_emb_pre = self.node_emb[(time_expand_s-1).type(LType), n_node.squeeze()].to(self.device)        # lsmooth = ((s_node_emb - s_node_emb_pre)**2).sum(dim=1) + ((t_node_emb - t_node_emb_pre)**2).sum(dim=1)

        p_static = (((s_node_emb - t_node_emb) ** 2).sum(dim=1))
        lstatic = t_weight * p_static

        # For HP loss
        delta = self.delta[s_node.squeeze()].to(self.device)
        base_p = -(((s_node_emb_pre - t_node_emb_pre) ** 2).sum(dim=1))

        hist_t_node_emb = self.node_emb[hist_t.type(LType), t_node.view(batch, -1).expand(batch, self.hist_len)]
        hist_s_node_emb = self.node_emb[hist_t.type(LType), s_node.view(batch, -1).expand(batch, self.hist_len)]
        hist_node_emb = self.node_emb[hist_t.view(batch, self.hist_len, 1).expand(batch, self.hist_len, self.hist_node), h_node]
        dist_ht = torch.abs(hist_node_emb.view(self.hist_node, batch, self.hist_len, -1) - hist_t_node_emb)
        dist_hs = torch.abs(hist_node_emb.view(self.hist_node, batch, self.hist_len, -1) - hist_s_node_emb)

        decay = delta.view(batch, -1) / (((t_time.view(batch, -1) - hist_t.type(FType).to(self.device).view(-1, self.hist_len))+0.01)**1.01)
        if self.attn == 1:
            attn_p = self.leakyrelu(
                torch.mm(torch.mm(torch.cat([dist_ht, dist_hs], dim=3).view(-1, 2 * self.emb_size), self.W),
                         self.a)).view(self.hist_node, batch, -1)
            attn_p = torch.softmax(attn_p, dim=0)
            exciting_p = (
                        (-1 * (dist_ht ** 2)).sum(dim=3) * attn_p * h_mask.view(self.hist_node, batch, -1) * decay).sum(
                dim=2).sum(dim=0)
        else:
            exciting_p = (
                        (-1 * (dist_ht ** 2)).sum(dim=3) * h_mask.view(self.hist_node, batch, -1) * decay).sum(
                dim=2).sum(dim=0)
        p_lambda = (base_p + exciting_p)

        base_n = -((n_node_emb_pre - s_node_emb_pre.view(batch, 1, self.emb_size)) ** 2).sum(dim=2)
        dist_hn = torch.abs((hist_node_emb.view(self.hist_node, self.hist_len, batch, 1, self.emb_size) - n_node_emb).view(-1, self.hist_node, batch, self.hist_len, self.emb_size))
        dist_hs_repeat = torch.abs(torch.cat(self.neg_size*[dist_hs]).view(-1, self.hist_node, batch, self.hist_len, self.emb_size))
        if self.attn == 1:
            b = torch.cat([dist_hs_repeat, dist_hn], dim=4).unsqueeze(4)
            attn_n = self.leakyrelu(
                torch.matmul(torch.matmul(b, self.W).squeeze(4),
                             self.a)).squeeze(4)
            attn_n = torch.softmax(attn_n, dim=1)
            exciting_n = (((-1 * (dist_hn ** 2)).sum(dim=4)) * attn_n * decay * h_mask.view(self.hist_node, batch,
                                                                                            -1)).sum(dim=3).sum(
                dim=1).view(batch, -1)
        else:
            exciting_n = (((-1 * (dist_hn ** 2)).sum(dim=4)) * decay * h_mask.view(self.hist_node, batch,
                                                                                            -1)).sum(dim=3).sum(
                dim=1).view(batch, -1)

        n_lambda = base_n + exciting_n
        loss_hp = -torch.log(torch.sigmoid(p_lambda) + 1e-6) - torch.log(
            torch.sigmoid(torch.neg(n_lambda)) + 1e-6).sum(dim=1)

        loss = torch.mean(loss_hp) + self.para_prox * torch.mean(lstatic) + self.para_smth * torch.mean(lsmooth)
        return loss

    def update(self, s_nodes, t_nodes, t_times, t_weight, n_nodes, h_node, h_mask, hist_t):
        with torch.cuda.device(DID):
            self.opt.zero_grad()
            loss = self.forward(s_nodes, t_nodes, t_times, t_weight, n_nodes, h_node, h_mask, hist_t)
            loss = loss.sum()
            self.loss += loss.data
            loss.backward()
            self.opt.step()

    def train(self):
        for epoch in range(self.epochs):
            self.loss = 0.0
            loader = DataLoader(self.data, batch_size=self.batch,
                                shuffle=True, num_workers=0)
            if epoch % self.save_step == 0 and epoch != 0:
                self.save_node_embeddings('./emb/facebook_neg_3/%d' % (epoch))
            sample_batched: object
            for i_batch, sample_batched in enumerate(loader):
                if i_batch % 100 == 0 and i_batch != 0:
                    sys.stdout.write('\r' + str(i_batch * self.batch) + '\tloss: ' + str(
                        self.loss.cpu().numpy() / (self.batch * i_batch)) + '\tdelta:' + str(
                        self.delta.mean().cpu().data.numpy()))
                    sys.stdout.flush()

                with torch.cuda.device(DID):
                    self.update(sample_batched['source_node'].type(LType).cuda(),
                                sample_batched['target_node'].type(LType).cuda(),
                                sample_batched['target_time'].type(FType).cuda(),
                                sample_batched['target_weight'].type(FType).cuda(),
                                sample_batched['neg_nodes'].type(LType).cuda(),
                                sample_batched['hist_node'].type(LType).cuda(),
                                sample_batched['hist_mask'].type(FType).cuda(),
                                sample_batched['hist_time'].type(LType).cuda(),
                                )

            sys.stdout.write('\repoch ' + str(epoch) + ': avg loss = ' +
                             str(self.loss.cpu().numpy() / len(self.data)) + '\n')
            sys.stdout.flush()

        self.save_node_embeddings('./emb/%d' % (self.epochs))

    def save_node_embeddings(self, path):
        if torch.cuda.is_available():
            embeddings = self.node_emb.cpu().data.numpy()
        else:
            embeddings = self.node_emb.data.numpy()
        for i in range(embeddings.shape[0]):
            if not os.path.exists(path):
                os.mkdir(path)
            emb_path = path + '/%d.emb' %i
            writer = open(emb_path, 'w')
            writer.write('%d %d\n' % (self.node_dim, self.emb_size))
            for n_idx in range(self.node_dim):
                writer.write(' '.join(str(d) for d in embeddings[i][n_idx]) + '\n')
            writer.close()


if __name__ == '__main__':
    hp_emb = HpEmb('./data/', directed=False)
    hp_emb.train()
