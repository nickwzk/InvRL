import numpy as np
import torch
import math


class Model:
    def __init__(self):
        self.val_scores = None
        self.test_scores = None
        self.val_ndcg = 0.0
        self.ds = None
        self.net = None
        self.logging = None

    def logscore(self, scores):
        self.logging.info('Precision %s' % str(scores[0]))
        self.logging.info('Recall %s' % str(scores[1]))
        self.logging.info('ndcg %s' % str(scores[2]))

    def test(self):
        P = []
        R = []
        N = []
        for i in [1, 5, 10]:
            precision, recall, ndcg_score = self.full_accuracy(self.ds.test, step=self.args.ssz, topk=i)
            P.append(precision)
            R.append(recall)
            N.append(ndcg_score)
        self.logging.info('----- test -----')
        self.test_scores = [P, R, N]
        self.logscore(self.test_scores)

    def val(self):
        P = []
        R = []
        N = []
        for i in [1, 5, 10]:
            precision, recall, ndcg_score = self.full_accuracy(self.ds.val, step=self.args.ssz, topk=i)
            P.append(precision)
            R.append(recall)
            N.append(ndcg_score)
        self.logging.info('----- val -----')
        self.val_scores = [P, R, N]
        self.val_ndcg = N[2]
        self.logscore(self.val_scores)

    def train(self):
        raise Exception('no implementation')

    def full_accuracy(self):
        raise Exception('no implementation')

    def predict(self):
        raise Exception('no implementation')

    def save(self):
        raise Exception('no implementation')

    def update(self):
        self.max_test = self.test_scores

    def full_accuracy(self, val_data, step=2000, topk=10):
        self.net.eval()
        start_index = 0
        end_index = self.ds.usz if step is None else step

        with torch.no_grad():
            all_index_of_rank_list = torch.LongTensor([])
            items = torch.LongTensor(range(self.ds.isz))
            while self.ds.usz >= end_index > start_index:
                users = torch.LongTensor(range(start_index, end_index))
                score_matrix = self.predict(users, items, True)

                for row, col in self.ds.user_item_dict.items():
                    if start_index <= row < end_index:
                        row -= start_index
                        col = torch.LongTensor(list(col)) - self.ds.usz
                        score_matrix[row][col] = 1e-5

                _, index_of_rank_list = torch.topk(score_matrix, topk)
                all_index_of_rank_list = torch.cat((all_index_of_rank_list, index_of_rank_list.cpu()), dim=0)
                start_index = end_index

                if end_index + step < self.ds.usz:
                    end_index += step
                else:
                    end_index = self.ds.usz

            length = len(val_data)
            precision = recall = ndcg = 0.0

            for data in val_data:
                user = data[0]
                pos_items = set(data[1:])
                num_pos = len(pos_items)
                if num_pos == 0:
                    length -= 1
                    continue
                items_list = all_index_of_rank_list[user].tolist()

                items = set(items_list)

                num_hit = len(pos_items.intersection(items))

                precision += float(num_hit / topk)
                if num_pos == 0:
                    self.logging.info(data)
                recall += float(num_hit / num_pos)

                ndcg_score = 0.0
                max_ndcg_score = 0.0

                for i in range(min(num_hit, topk)):
                    max_ndcg_score += 1 / math.log2(i + 2)
                if max_ndcg_score == 0:
                    continue

                for i, temp_item in enumerate(items_list):
                    if temp_item in pos_items:
                        ndcg_score += 1 / math.log2(i + 2)

                ndcg += ndcg_score / max_ndcg_score

        return precision / length, recall / length, ndcg / length
