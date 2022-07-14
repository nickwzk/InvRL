# UltraGCN

import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
from model import Model
import scipy.sparse as sp
import math


class UltraGCNNet(torch.nn.Module):
    def __init__(self, ds, args, logging, mask=None, has_bias=True):
        super().__init__()
        self.ds = ds
        self.args = args
        self.logging = logging
        self.has_bias = has_bias

        if mask is None or mask.shape[0] < 5:
            self.mask = torch.ones(self.ds.feature.shape[1]).to(self.args.device)
        else:
            self.mask = torch.FloatTensor(mask).to(self.args.device)

        self.U = torch.nn.Parameter(torch.rand(self.ds.usz, self.ds.dim + self.args.feat_dim), requires_grad=True)
        torch.nn.init.normal_(self.U, mean=0.0, std=0.01)
        self.V = torch.nn.Parameter(torch.rand(self.ds.isz, self.ds.dim), requires_grad=True)
        torch.nn.init.normal_(self.V, mean=0.0, std=0.01)

        self.num_modal = 0
        self.v_feat = None
        self.a_feat = None
        self.t_feat = None
        if self.ds.v_feat is not None:
            self.v_feat = self.ds.v_feat.to(self.args.device)
            self.v_feat = torch.nn.functional.normalize(self.v_feat, dim=1)
            self.num_modal += 1
        if self.ds.a_feat is not None:
            self.a_feat = self.ds.a_feat.to(self.args.device)
            self.a_feat = torch.nn.functional.normalize(self.a_feat, dim=1)
            self.num_modal += 1
        if self.ds.t_feat is not None:
            self.t_feat = self.ds.t_feat.to(self.args.device)
            self.t_feat = torch.nn.functional.normalize(self.t_feat, dim=1)
            self.num_modal += 1
        self.MLP = torch.nn.Linear(self.ds.feature.shape[1], self.args.feat_dim, bias=has_bias)
        torch.nn.init.normal_(self.MLP.weight, mean=0.0, std=0.01)
        if self.has_bias:
            torch.nn.init.constant_(self.MLP.bias, 0)

        self.emb_params = [self.U] + [self.V]
        self.proj_params = list(self.MLP.parameters())

        self.constraint_mat = None
        self.pre()

        self.user_emb = None
        self.item_emb = None
        self.word_emb = None
        self.num_word = None
        self.word_mat = None

        self.w1, self.w2, self.w3, self.w4 = self.args.p_w

        if self.args.dataset == 'tiktok':
            self.init_feat()

    def init_feat(self):
        self.num_word = torch.max(self.ds.t_data[1]) + 1
        self.word_emb = torch.nn.Parameter(torch.rand(self.num_word, 128), requires_grad=True)
        torch.nn.init.normal_(self.word_emb, mean=0.0, std=0.01)
        pos = torch.LongTensor(self.ds.t_data)
        val = torch.ones(self.ds.t_data.shape[1])
        self.word_mat = torch.sparse_coo_tensor(pos, val, (self.ds.isz, self.num_word)).to(self.args.device)

        self.emb_params += [self.word_emb]

    def pre(self):
        train_mat = sp.dok_matrix((self.ds.usz, self.ds.isz), dtype=np.float32)

        for x in self.ds.train:
            train_mat[x[0], x[1]] = 1.0

        D_u = np.sum(train_mat, axis=1).reshape(-1)
        D_i = np.sum(train_mat, axis=0).reshape(-1)

        epsilon = 1e-10
        D_u[D_u < epsilon] = epsilon

        beta_u = (np.sqrt(D_u + 1) / D_u).reshape(-1, 1)
        beta_i = (1 / np.sqrt(D_i + 1)).reshape(1, -1)
        self.constraint_mat = {"beta_u": torch.from_numpy(beta_u).reshape(-1).to(self.args.device), "beta_i": torch.from_numpy(beta_i).reshape(-1).to(self.args.device)}

    def cal_weight(self, uid, iid, niid):
        pos_weight = torch.mul(self.constraint_mat['beta_u'][uid], self.constraint_mat['beta_i'][iid])
        pos_weight = self.w1 + self.w2 * pos_weight

        neg_weight = torch.mul(torch.repeat_interleave(self.constraint_mat['beta_u'][uid], niid.size(1)), self.constraint_mat['beta_i'][niid.flatten()])
        neg_weight = self.w3 + self.w4 * neg_weight

        return pos_weight, neg_weight

    def loss_L(self, uid, iid, niid):
        beta_p, beta_n = self.cal_weight(uid, iid, niid)

        pred_p = self.predict(uid, iid)
        pred_n = self.predict(uid, niid, True)
        label_p = torch.ones(pred_p.size()).to(self.args.device)
        label_n = torch.zeros(pred_n.size()).to(self.args.device)

        loss_p = F.binary_cross_entropy_with_logits(pred_p, label_p, weight=beta_p, reduction='sum')
        loss_n = F.binary_cross_entropy_with_logits(pred_n, label_n, weight=beta_n.view(pred_n.size()), reduction='none').mean(dim=-1).sum()

        loss = loss_p + loss_n

        return loss

    def regs(self, uid, iid, niid):
        lr1, wd1 = self.args.p_emb
        lr2, wd2 = self.args.p_proj
        emb_regs = 0.0
        proj_regs = 0.0
        emb_regs += torch.sum(self.U ** 2)
        emb_regs += torch.sum(self.V ** 2) * self.args.wdi
        if self.args.dataset == 'tiktok':
            emb_regs += torch.sum(self.word_emb ** 2)
        proj_regs += torch.sum(self.MLP.weight ** 2)
        if self.has_bias:
            proj_regs += torch.sum(self.MLP.bias ** 2)
        return wd1 * emb_regs + wd2 * proj_regs

    def cal_t(self):
        self.t_feat = torch.matmul(self.word_mat, self.word_emb)
        self.t_feat = torch.nn.functional.normalize(self.t_feat, dim=1)

    def forward(self, uid, iid, niid, fs=None):
        if self.args.dataset == 'tiktok':
            self.cal_t()

        self.user_emb = self.U

        feat = torch.Tensor([]).to(self.args.device)
        if self.v_feat is not None:
            feat = torch.cat((feat, self.v_feat), dim=1)
        if self.a_feat is not None:
            feat = torch.cat((feat, self.a_feat), dim=1)
        if self.t_feat is not None:
            feat = torch.cat((feat, self.t_feat), dim=1)
        feat = feat * self.mask
        if fs is not None:
            feat = fs(feat)
        feat = self.MLP(feat)
        self.item_emb = torch.cat((self.V, feat), dim=1)

        loss = self.loss_L(uid, iid, niid) + self.regs(uid, iid, niid)
        return loss

    def predict(self, uid, iid, flag=False):
        if self.user_emb is None:
            return None
        if flag:
            return torch.sum(self.user_emb[uid].unsqueeze(1) * self.item_emb[iid], dim=2)
        return torch.sum(self.user_emb[uid] * self.item_emb[iid], dim=1)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class UltraGCN(Model):
    def __init__(self, ds, args, logging):
        super().__init__()
        setup_seed(2233)
        self.filename = 'weights/%s_UGCN_best.pth' % args.dataset
        self.ds = ds
        self.args = args
        self.logging = logging
        self.args.wdi = 1

        self.net = UltraGCNNet(self.ds, self.args, self.logging).to(self.args.device)

        self.max_test = None
        self.max_net = None

    def predict(self, uid, iid, flag=False):
        return self.net.predict(uid, iid, flag)

    def update(self):
        self.max_test = self.test_scores
        torch.save(self.net.state_dict(), self.filename)

    def train(self):
        lr1, wd1 = self.args.p_emb
        lr2, wd2 = self.args.p_proj
        optimizer = torch.optim.Adam(self.net.emb_params, lr=lr1, weight_decay=0)
        optimizer2 = torch.optim.Adam(self.net.proj_params, lr=lr2, weight_decay=0)

        temp_id = torch.zeros(self.args.bsz, 2).type(torch.int64)
        temp_nid = torch.zeros(self.args.bsz, self.args.neg_num).type(torch.int64)
        self.net(temp_id[:, 0], temp_id[:, 1], temp_nid)

        epochs = self.args.num_epoch
        val_max = 0.0
        num_decreases = 0
        max_epoch = 0
        end_epoch = epochs
        for epoch in tqdm(range(epochs)):
            generator = self.ds.sample()
            while True:
                self.net.train()
                optimizer.zero_grad()
                optimizer2.zero_grad()
                uid, iid, niid = next(generator)
                if uid is None:
                    break
                uid, iid, niid = uid.to(self.args.device), iid.to(self.args.device), niid.to(self.args.device)

                loss = self.net(uid, iid, niid)

                loss.backward()
                optimizer.step()
                optimizer2.step()

            if epoch > 0 and epoch % self.args.epoch == 0:
                self.logging.info("Epoch %d: loss %s, U.norm %s, V.norm %s, MLP.norm %s" % (epoch, loss, torch.norm(self.net.U).item(), torch.norm(self.net.V).item(), torch.norm(self.net.MLP.weight).item()))
                self.val(), self.test()
                if self.val_ndcg > val_max:
                    val_max = self.val_ndcg
                    max_epoch = epoch
                    num_decreases = 0
                    self.update()
                else:
                    if num_decreases > 10:
                        end_epoch = epoch
                        break
                    else:
                        num_decreases += 1

        self.logging.info("Epoch %d:" % end_epoch)
        self.val(), self.test()
        if self.val_ndcg > val_max:
            val_max = self.val_ndcg
            max_epoch = epochs
            num_decreases = 0
            self.update()

        self.logging.info("final:")
        self.logging.info('----- test -----')
        self.logscore(self.max_test)
        self.logging.info('max_epoch %d:' % max_epoch)
