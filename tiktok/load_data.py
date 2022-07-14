import numpy as np
import torch
import torch.utils.data as D
# from sklearn.decomposition import PCA


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    # torch.backends.cudnn.deterministic = True


class dataset:
    def __init__(self, logging, args):
        self.args = args
        self.logging = logging
        setup_seed(2233)
        # np.random.seed(1122)

        self.name = 'tiktok'
        self.train = np.load('tiktok/train.npy', allow_pickle=True)
        self.val = np.load('tiktok/val_full.npy', allow_pickle=True)
        self.test = np.load('tiktok/test_full.npy', allow_pickle=True)
        self.user_item_dict = np.load('tiktok/user_item_dict.npy', allow_pickle=True).item()

        self.usz = np.int64(36656)
        self.train[:, 1] -= self.usz
        self.isz = np.int64(76085)
        self.dim = 64
        self.bsz = self.args.bsz
        self.sz = self.train.shape[0]

        # val & test
        for data in self.val:
            data[1:] -= self.usz

        for data in self.test:
            data[1:] -= self.usz

        # self.train_loader = D.DataLoader(self.train, batch_size=self.bsz, shuffle=True)

        # feature
        # v_feat = np.load('tiktok/feat_v.npy', allow_pickle=True).astype(np.float32)
        # a_feat = np.load('tiktok/feat_a.npy', allow_pickle=True).astype(np.float32)
        # t_feat = np.load('tiktok/feat_t.npy', allow_pickle=True).astype(np.float32)
        self.v_feat = torch.load('tiktok/feat_v.pt').type(torch.float32)
        self.a_feat = torch.load('tiktok/feat_a.pt').type(torch.float32)
        self.t_feat = torch.zeros(self.isz, 128)
        self.t_data = torch.load('tiktok/feat_t.pt')

        self.feature = torch.cat((self.v_feat, self.a_feat, self.t_feat), dim=1)
        # feature = np.concatenate([v_feat, a_feat, t_feat], axis=1)
        self.logging.info(self.feature.shape)

    def sample_neg(self, bsz):
        neg_candidates = np.arange(self.isz)
        neg_items = np.random.choice(neg_candidates, (bsz, self.args.neg_num), replace=True)
        neg_items = torch.from_numpy(neg_items)
        return neg_items

    def sample(self, domain=None, current_domain=None):
        if current_domain is None:
            np.random.shuffle(self.train)

            start_index = 0
            end_index = self.args.bsz
            if end_index >= self.sz:
                end_index = self.sz
            while start_index < end_index <= self.sz:
                sub_train_list = torch.from_numpy(self.train[start_index: end_index, :])
                neg_items = self.sample_neg(end_index - start_index)
                start_index = end_index
                end_index += self.args.bsz
                if end_index >= self.sz:
                    end_index = self.sz

                yield sub_train_list[:, 0], sub_train_list[:, 1], neg_items
            yield None, None, None
        else:
            temp_pos = torch.where(domain == current_domain)[0]
            temp_pos = temp_pos.cpu().numpy()
            temp_train = self.train[temp_pos, :]
            np.random.shuffle(temp_train)
            sz = temp_train.shape[0]

            start_index = 0
            end_index = self.args.bsz
            if end_index >= sz:
                end_index = sz
            while start_index < end_index <= sz:
                sub_train_list = torch.from_numpy(temp_train[start_index: end_index, :])
                neg_items = self.sample_neg(end_index - start_index)
                start_index = end_index
                end_index += self.args.bsz
                if end_index >= sz:
                    end_index = sz

                yield sub_train_list[:, 0], sub_train_list[:, 1], neg_items
            yield None, None, None
