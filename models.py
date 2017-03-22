from collections import defaultdict
import random


def train_test_set(data):
    test_users = set(random.sample(list(data['user_id'].unique()), int(len(data['user_id'].unique()) * 0.2)))
    data_test = data[data['user_id'].isin(test_users)]
    data_train = data[~data['user_id'].isin(test_users)]
    return data_train, data_test


class ItemAverage:

    def __init__(self):
        self._nums = defaultdict(lambda: 0)
        self._corrects = defaultdict(lambda: 0)

    def train(self, trainset):
        for p_recall, lexeme_id in trainset[['p_recall', 'lexeme_id']].values:
            self._nums[lexeme_id] += 1
            self._corrects[lexeme_id] += p_recall

    def predict(self, lexeme_id):
        if lexeme_id not in self._nums:
            return 0.5
        return self._corrects[lexeme_id] / self._nums[lexeme_id]
