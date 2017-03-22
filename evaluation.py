from sklearn import metrics
import math
import matplotlib.pyplot as plt
import numpy


def rmse(predictions, real):
    return math.sqrt(numpy.mean((numpy.array(predictions) - numpy.array(real)) ** 2))


def mae(predictions, real):
    return numpy.mean([abs(p - r) for p, r in zip(predictions, real)])


def load_brier(predictions, real, bins=20):
    counts = numpy.zeros(bins)
    correct = numpy.zeros(bins)
    prediction = numpy.zeros(bins)
    for p, r in zip(predictions, real):
        bin = min(int(p * bins), bins - 1)
        counts[bin] += 1
        correct[bin] += r
        prediction[bin] += p
    prediction_means = prediction / counts
    prediction_means[numpy.isnan(prediction_means)] = ((numpy.arange(bins) + 0.5) / bins)[numpy.isnan(prediction_means)]
    correct_means = correct / counts
    correct_means[numpy.isnan(correct_means)] = 0
    size = len(predictions)
    answer_mean = sum(correct) / size
    return {
        "reliability": sum(counts * (correct_means - prediction_means) ** 2) / size,
        "resolution": sum(counts * (correct_means - answer_mean) ** 2) / size,
        "uncertainty": answer_mean * (1 - answer_mean),
        "detail": {
            "bin_count": bins,
            "bin_counts": list(counts),
            "bin_prediction_means": list(prediction_means),
            "bin_correct_means": list(correct_means),
        }
    }


def plot_model_stats(predicted, observed, bins=20):
    print("RMSE:", rmse(predicted, observed))
    print("MAE:", mae(predicted, observed))
    print("AUC:", metrics.roc_auc_score(numpy.array(observed) > 0.5, predicted))
    plot_brier(predicted, observed, bins)
    plt.show()


def plot_brier(predictions, real, bins=20):
    brier = load_brier(predictions, real, bins=bins)
    plt.figure()
    plt.plot(brier['detail']['bin_prediction_means'], brier['detail']['bin_correct_means'], label='Average observation')
    plt.plot((0, 1), (0, 1), label='Optimal average observation')
    bin_count = brier['detail']['bin_count']
    counts = numpy.array(brier['detail']['bin_counts'])
    bins = (numpy.arange(bin_count) + 0.5) / bin_count
    plt.legend(loc='upper center')
    plt.xlabel('Prediction')
    plt.ylabel('Observeation')
    plt.twinx()
    plt.ylabel('Number of predictions')
    plt.bar(bins, counts, width=(0.5 / bin_count), alpha=0.5, label='Number of predictions')
    plt.legend(loc='lower center')
