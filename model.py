############################################################
# This code builds on https://github.com/CRIPAC-DIG/TAGNN #
############################################################

from tqdm import tqdm
import datetime
import math
import numpy as np

import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F

from agc import AGC


class Attention_GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(Attention_GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]],
                                self.linear_edge_in(hidden)) + self.b_iah

        input_out = torch.matmul(
            A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah

        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden


class Attention_SessionGraph(Module):
    def __init__(self, opt, n_node):
        super(Attention_SessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.tagnn = Attention_GNN(self.hidden_size, step=opt.step)

        self.layer_norm1 = nn.LayerNorm(self.hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size, num_heads=2, dropout=0.1)

        self.linear_one = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)

        self.linear_two = nn.Linear(
            self.hidden_size, self.hidden_size, bias=True)

        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(
            self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear_t = nn.Linear(
            self.hidden_size, self.hidden_size, bias=False)  # target attention
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.agc_optimizer = AGC(self.parameters(), self.optimizer, model=self)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_scores(self, hidden, mask):
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(
            mask, 1) - 1]  # batch_size x latent_size
        # batch_size x 1 x latent_size
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        # batch_size x seq_length x 1
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        alpha = F.softmax(alpha, 1)  # batch_size x seq_length x 1
        # batch_size x latent_size
        a = torch.sum(alpha * hidden *
                      mask.view(mask.shape[0], -1, 1).float(), 1)

        if not self.nonhybrid:
            a = self.linear_transform(torch.cat([a, ht], 1))
        b = self.embedding.weight[1:]  # n_nodes x latent_size

        # batch_size x seq_length x latent_size
        hidden = hidden * mask.view(mask.shape[0], -1, 1).float()
        qt = self.linear_t(hidden)  # batch_size x seq_length x latent_size
        # batch_size x n_nodes x seq_length
        beta = F.softmax(b @ qt.transpose(1, 2), -1)
        target = beta @ hidden  # batch_size x n_nodes x latent_size
        a = a.view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        a = a + target  # batch_size x n_nodes x latent_size
        scores = torch.sum(a * b, -1)  # batch_size x n_nodes
        return scores

    def forward(self, inputs, A):
        hidden = self.embedding(inputs)
        hidden = self.tagnn(A, hidden)
        hidden = hidden.permute(1, 0, 2)

        skip = self.layer_norm1(hidden)
        hidden, attn_w = self.attn(
            hidden, hidden, hidden, attn_mask=get_mask(hidden.shape[0]))
        hidden = hidden+skip
        hidden = hidden.permute(1, 0, 2)

        return hidden


def get_mask(seq_len):
    return torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1).astype('bool')).to('cuda')


def to_cuda(input_variable):
    if torch.cuda.is_available():
        return input_variable.cuda()
    else:
        return input_variable


def to_cpu(input_variable):
    if torch.cuda.is_available():
        return input_variable.cpu()
    else:
        return input_variable


def forward(model, i, data):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = to_cuda(torch.Tensor(alias_inputs).long())
    items = to_cuda(torch.Tensor(items).long())
    A = to_cuda(torch.Tensor(A).float())
    mask = to_cuda(torch.Tensor(mask).long())
    hidden = model(items, A)

    def get(i): return hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i)
                             for i in torch.arange(len(alias_inputs)).long()])

    return targets, model.compute_scores(seq_hidden, mask)


def train_test(model, train_data, test_data):
    model.scheduler.step()
    print('Start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)

    for i, j in tqdm(zip(slices, np.arange(len(slices))), total=len(slices)):
        model.optimizer.zero_grad()
        targets, scores = forward(model, i, train_data)
        targets = to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()

        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))

    print('\tLoss Value:\t%.3f' % total_loss)
    print('Start Prediction: ', datetime.datetime.now())

    model.eval()
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)

    for i in slices:
        targets, scores = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = to_cpu(sub_scores).detach().numpy()

        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))

    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr


def get_pos(seq_len):
    return torch.arange(seq_len).unsqueeze(0)
