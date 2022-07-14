import numpy as np
import torch
from tqdm import tqdm
import torch.nn.functional as F
# from util import Metric
from model import Model
import scipy.sparse as sp
import math
from torch.autograd import grad
from UltraGCN import UltraGCNNet


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


class FrontModel(torch.nn.Module):
    def __init__(self, ds, args, logging):
        super().__init__()
        setup_seed(2233)
        self.ds = ds
        self.args = args
        self.logging = logging
        self.filename_pre = 'weights/%s_UGCN_best.pth' % args.dataset

        self.net = UltraGCNNet(self.ds, self.args, self.logging).to(self.args.device)

        self.weight = None

    def predict(self, uid, iid, flag=False):
        return self.net.predict(uid, iid, flag)

    def reg_loss(self):
        lr2, wd2 = self.args.p_proj
        loss = torch.mean(torch.abs(self.net.MLP.weight * self.weight))
        return wd2 * loss

    def init_frontmodel(self):
        self.net.load_state_dict(torch.load(self.filename_pre), strict=False)
        for p in self.net.parameters():
            p.requires_grad = False
        torch.nn.init.normal_(self.net.MLP.weight, mean=0.0, std=0.01)
        torch.nn.init.constant_(self.net.MLP.bias, 0)
        self.net.MLP.weight.requires_grad = True
        self.net.MLP.bias.requires_grad = True

    def train(self, m_weight, domain, current_domain):
        self.weight = m_weight
        lr1, wd1 = self.args.p_emb
        lr2, wd2 = self.args.p_proj
        optimizer = torch.optim.Adagrad(self.net.emb_params, lr=lr1, weight_decay=0)
        optimizer2 = torch.optim.Adam(self.net.proj_params, lr=lr2, weight_decay=0)

        epochs = self.args.f_epoch
        for epoch in tqdm(range(epochs)):
            generator = self.ds.sample(domain, current_domain)
            loss_sum = 0.0
            while True:
                self.net.train()
                optimizer.zero_grad()
                optimizer2.zero_grad()
                uid, iid, niid = next(generator)
                if uid is None:
                    break
                uid, iid, niid = uid.to(self.args.device), iid.to(self.args.device), niid.to(self.args.device)

                loss = self.net(uid, iid, niid) + self.reg_loss()
                loss_sum += loss.detach()

                loss.backward()
                optimizer.step()
                optimizer2.step()

            if epoch > 0 and (epoch + 1) % self.args.epoch == 0:
                self.logging.info("Epoch %d: loss %s, U.norm %s, V.norm %s, MLP.norm %s" % (epoch + 1, loss_sum, torch.norm(self.net.U).item(), torch.norm(self.net.V).item(), torch.norm(self.net.MLP.weight).item()))


class FeatureSelector(torch.nn.Module):
    def __init__(self, input_dim, sigma, args):
        super().__init__()
        setup_seed(2233)
        self.args = args
        self.mu = torch.nn.Parameter(torch.zeros(input_dim), requires_grad=True)
        self.noise = torch.randn(self.mu.size()).to(self.args.device)
        self.sigma = sigma
        self.input_dim = input_dim

    def renew(self):
        torch.nn.init.zeros_(self.mu)
        self.noise = torch.randn(self.mu.size()).to(self.args.device)

    def forward(self, prev_x):
        z = self.mu + self.sigma * self.noise.normal_() * self.training
        stochastic_gate = self.hard_sigmoid(z)
        new_x = prev_x * stochastic_gate
        return new_x

    def hard_sigmoid(self, x):
        return torch.clamp(x + 0.5, 0.0, 1.0)

    def reg(self, x):
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


class InvRL(Model):
    def __init__(self, ds, args, logging):
        super().__init__()
        setup_seed(2233)
        self.filename_pre = 'weights/%s_UGCN_best.pth' % args.dataset
        self.filename = 'weights/%s_InvRL_best.pth' % args.dataset
        self.ds = ds
        self.args = args
        self.logging = logging

        self.max_test = None
        self.max_net = None

        self.mask_dim = self.ds.feature.shape[1]
        self.domain = torch.tensor(np.random.randint(0, self.args.num_domains, self.ds.train.shape[0])).to(self.args.device)
        self.weight = torch.tensor(np.zeros(self.mask_dim, dtype=np.float32)).to(self.args.device)
        self.proj = None

        self.net_e = None

        self.fs = FeatureSelector(self.mask_dim, self.args.sigma, args).to(self.args.device)
        self.lam = self.args.lam
        self.alpha = self.args.alpha
        self.backmodel = UltraGCNNet(self.ds, self.args, self.logging, has_bias=False).to(self.args.device)

        self.net = None

    def cal_error(self, net_id):
        with torch.no_grad():
            result = torch.Tensor([]).to(self.args.device)
            start_index = 0
            end_index = self.args.ssz
            while start_index < end_index <= self.ds.sz:
                sub = torch.from_numpy(self.ds.train[start_index: end_index, :]).to(self.args.device)
                pred = self.net_e.predict(sub[:, 0], sub[:, 1])
                if pred is None:
                    pred = torch.zeros_like(sub[:, 0])
                result = torch.cat((result, pred), dim=0)
                start_index = end_index
                end_index += self.args.ssz
                if end_index >= self.ds.sz:
                    end_index = self.ds.sz
            return result

    def init_word_emb(self, net):
        word_emb = torch.load(self.filename_pre)['word_emb']
        net.word_emb.data.copy_(word_emb)
        net.word_emb.requires_grad = False

    def frontend(self):
        self.logging.info('----- frontend -----')
        ite = 0
        delta_threshold = int(self.ds.train.shape[0] * 0.01)
        print('delta_threshold %d' % delta_threshold)
        if self.args.reuse == 0:
            self.domain = torch.tensor(np.random.randint(0, self.args.num_domains, self.ds.train.shape[0])).to(
                self.args.device)
            print('domain :', self.domain)

        while True:
            ite += 1
            self.net_e = None
            tot_result = []
            for i in range(self.args.num_domains):
                self.logging.info('Environment %d' % i)
                self.net_e = FrontModel(self.ds, self.args, self.logging).to(self.args.device)
                if self.args.dataset == 'tiktok':
                    self.init_word_emb(self.net_e.net)
                self.net_e.train(self.weight, self.domain, i)
                result = self.cal_error(i)
                tot_result.append(result)

            tot_result = torch.stack(tot_result, dim=0)
            new_domain = torch.argmax(tot_result, dim=0)
            diff = self.domain.reshape(-1, 1) - new_domain.reshape(-1, 1)
            diff[diff != 0] = 1
            delta = int(torch.sum(diff))
            print('Ite = %d, Delta = %d' % (ite, delta))
            self.logging.info('Ite = %d, Delta = %d' % (ite, delta))
            self.domain = new_domain
            if delta <= delta_threshold or ite >= self.args.f_max:
                break

        print(self.domain)
        self.net_e = None

    def predict(self, uid, iid, flag=False):
        return self.net.predict(uid, iid, flag)

    def single_forward(self, uid, iid, niid):
        assert self.fs.training is True
        loss_single = self.backmodel(uid, iid, niid, self.fs)
        grad_single = grad(loss_single, self.backmodel.MLP.weight, create_graph=True)[0]
        return loss_single, grad_single

    def loss_p(self, loss_avg, grad_avg, grad_list):
        penalty = torch.zeros_like(grad_avg).to(self.args.device)
        for gradient in grad_list:
            penalty += (gradient - grad_avg) ** 2
        penalty_detach = torch.sum((penalty * (self.fs.mu + 0.5)) ** 2)
        reg = self.fs.reg((self.fs.mu + 0.5) / self.fs.sigma)
        reg_penalty = torch.sum(self.fs.mu ** 2)
        total_loss = loss_avg + self.alpha * penalty_detach
        total_loss = total_loss + self.lam * reg_penalty
        return total_loss, penalty_detach, reg

    def init_backmodel(self):
        self.backmodel.load_state_dict(torch.load(self.filename_pre), strict=False)
        self.fs.renew()
        for p in self.backmodel.parameters():
            p.requires_grad = False
        torch.nn.init.normal_(self.backmodel.MLP.weight, mean=0.0, std=0.01)
        self.backmodel.MLP.weight.requires_grad = True

    def backend(self):
        self.logging.info('----- backend -----')
        self.init_backmodel()
        lr1, wd1 = self.args.p_emb
        lr2, wd2 = self.args.p_proj
        optimizer2 = torch.optim.Adam([{'params': self.backmodel.proj_params, 'lr': lr2, 'weight_decay': 0}, {'params': self.fs.mu, 'lr': self.args.lr, 'weight_decay': 0}])

        epochs = self.args.b_epoch
        reg = None

        for epoch in tqdm(range(epochs)):
            generator = []
            for i in range(self.args.num_domains):
                generator.append(self.ds.sample(self.domain, i))
            end_flag = False
            finish = [0 for i in range(self.args.num_domains)]
            loss = 0.0
            while end_flag is False:
                self.backmodel.train()
                self.fs.train()
                optimizer2.zero_grad()
                loss_avg = 0.0
                grad_avg = torch.zeros_like(self.backmodel.MLP.weight).to(self.args.device)  # 0.0
                grad_list = []
                for i in range(self.args.num_domains):
                    uid, iid, niid = next(generator[i])
                    if uid is None:
                        finish[i] = 1
                        if sum(finish) < self.args.num_domains:
                            generator[i] = self.ds.sample(self.domain, i)
                            uid, iid, niid = next(generator[i])
                        else:
                            end_flag = True
                            break
                    if uid is None:
                        continue
                    uid, iid, niid = uid.to(self.args.device), iid.to(self.args.device), niid.to(self.args.device)
                    loss_single, grad_single = self.single_forward(uid, iid, niid)
                    assert loss_single >= 0
                    loss_avg += loss_single / self.args.num_domains
                    grad_avg += grad_single / self.args.num_domains
                    grad_list.append(grad_single)

                loss, penalty, reg = self.loss_p(loss_avg, grad_avg, grad_list)
                loss.backward()
                optimizer2.step()

            if epoch > 0 and (epoch + 1) % self.args.epoch == 0:
                self.logging.info("Epoch %d: loss %s, reg %s mu %s MLP.norm %s" % (epoch + 1, loss, reg, self.fs.mu, torch.norm(self.backmodel.MLP.weight)))

        self.proj = self.backmodel.MLP.weight.detach()

        return self.fs.hard_sigmoid(self.fs.mu).detach(), reg.detach()

    def solve(self, ite=3):
        for i in range(ite):
            self.frontend()
            weight, density = self.backend()
            self.weight = weight
            self.lam *= 1.05
            self.alpha *= 1.05

        self.backmodel = None

    def train(self):
        if self.args.pretrained == 0:
            self.solve(self.args.ite)
            mask = self.weight
        else:
            mask = np.load(self.mask_filename, allow_pickle=True)
            mask = torch.from_numpy(mask)
        self.logging.info('mask %s' % mask)
        self.args.p_emb = self.args.p_embp
        self.args.p_proj = self.args.p_ctx
        self.net = UltraGCNNet(self.ds, self.args, self.logging, mask.cpu()).to(self.args.device)

        if self.args.dataset == 'tiktok':
            self.init_word_emb(self.net)

        lr1, wd1 = self.args.p_emb
        lr2, wd2 = self.args.p_proj
        optimizer = torch.optim.Adam(self.net.emb_params, lr=lr1, weight_decay=0)
        optimizer2 = torch.optim.Adam(self.net.proj_params, lr=lr2, weight_decay=0)

        epochs = self.args.num_epoch
        val_max = 0.0
        num_decreases = 0
        max_epoch = 0
        end_epoch = epochs
        loss = 0.0
        self.fs.eval()
        assert self.fs.training is False

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
                    if num_decreases > 40:
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
