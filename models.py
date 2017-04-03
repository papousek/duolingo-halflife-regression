from collections import defaultdict
import numpy as np
import random

def ZERO():
    return 0


def train_test_set(data):
    test_users = set(random.sample(list(data['user_id'].unique()), int(len(data['user_id'].unique()) * 0.2)))
    data_test = data[data['user_id'].isin(test_users)]
    data_train = data[~data['user_id'].isin(test_users)]
    return data_train, data_test


class Model:

    @staticmethod
    def from_name(model_name, *args, **kwargs):
        for model_class in Model._transitive_subclasses():
            if model_class.__name__ == model_name:
                return model_class(*args, **kwargs)
        raise Exception('There is no model with name {}.'.format(model_name))

    @staticmethod
    def _transitive_subclasses(given_class=None):
        if given_class is None:
            given_class = Model
        here = given_class.__subclasses__()
        return here + [sc for scs in [Model._transitive_subclasses(c) for c in here] for sc in scs]


class ItemAverage(Model):

    def __init__(self):
        self._nums = defaultdict(ZERO)
        self._corrects = defaultdict(ZERO)

    def train(self, trainset):
        predicted = np.zeros(len(trainset))
        for i, (p_recall, lexeme_id) in enumerate(trainset[['p_recall', 'lexeme_id']].values):
            predicted[i] = self.predict(lexeme_id)
            self._nums[lexeme_id] += 1
            self._corrects[lexeme_id] += p_recall
        return predicted

    def predict(self, lexeme_id):
        if lexeme_id not in self._nums:
            return 0.5
        return self._corrects[lexeme_id] / self._nums[lexeme_id]


class Elo(Model):

    def __init__(self, alpha=1.1, beta=0.09):
        self._skill = defaultdict(ZERO)
        self._difficulty = defaultdict(ZERO)
        self._count = defaultdict(ZERO)
        self._alpha = alpha
        self._beta = beta

    def train(self, trainset):
        predicted = np.zeros(len(trainset))
        for i, (p_recall, lexeme_id, user_id) in enumerate(trainset[['p_recall', 'lexeme_id', 'user_id']].values):
            predicted[i] = self.predict(user_id, lexeme_id, p_recall)
        return predicted

    def predict(self, user_id, lexeme_id, p_recall=None):
        prediction = _sigmoid(self._skill[user_id] - self._difficulty[lexeme_id])
        if p_recall is not None and (user_id, lexeme_id) not in self._count:
            item_alpha = self._alpha / (1 + self._beta * self._count[lexeme_id])
            user_alpha = self._alpha / (1 + self._beta * self._count[user_id])
            self._count[user_id] += 1
            self._count[lexeme_id] += 1
            self._count[user_id, lexeme_id] = 1
            self._difficulty[lexeme_id] -= item_alpha * (p_recall - prediction)
            self._skill[user_id] += user_alpha * (p_recall - prediction)
        return prediction

    def difficulty(self, lexeme_id):
        return self._difficulty[lexeme_id]

    def skill(self, user_id):
        return self._skill[user_id]


def _sigmoid(x):
    return 1.0 / (1 + np.exp(-x))
