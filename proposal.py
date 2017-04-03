from evaluation import rmse
from models import train_test_set, Model
from spiderpig import cached
import numpy as np
import pandas as pd


DAY_SECONDS = 60 * 60 * 24


@cached()
def load_traces(traces_filename='./data/settles.acl16.learning_traces.13m.csv.gz', traces_nrows=None):
    traces = pd.read_csv(traces_filename, nrows=traces_nrows)
    traces['delta_days'] = traces['delta'].apply(lambda d: d / DAY_SECONDS)
    return traces


@cached()
def load_train_test_set():
    return train_test_set(load_traces())


@cached()
def train_model(class_name, *args, **kwargs):
    model = Model.from_name(class_name, *args, **kwargs)
    trainset, _ = load_train_test_set()
    predicted = model.train(trainset)
    return model, predicted


@cached()
def train_model_rmse(class_name, *args, **kwargs):
    trainset, _ = load_train_test_set()
    _, predicted = train_model(class_name, *args, **kwargs)
    return rmse(predicted, trainset['p_recall'])


@cached()
def grid_search(model_name, param1, param2, param1_size=10, param2_size=10):
    name1, bounds1 = param1
    name2, bounds2 = param2
    vals1 = np.linspace(bounds1[0], bounds1[1], param1_size + 1)
    vals2 = np.linspace(bounds2[0], bounds2[1], param2_size + 1)
    combs = np.transpose([np.tile(vals1, len(vals2)), np.repeat(vals2, len(vals1))])
    result = []
    for val1, val2 in combs:
        result.append({
            name1: val1,
            name2: val2,
            'rmse': train_model_rmse(model_name, **{name1: val1, name2: val2}),
        })
    return pd.DataFrame(result)
