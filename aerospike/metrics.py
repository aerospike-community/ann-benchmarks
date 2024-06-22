from __future__ import absolute_import

import numpy as np


def knn_threshold(data, count, epsilon):
    return data[count - 1] + epsilon


def epsilon_threshold(data, count, epsilon):
    return data[count - 1] * (1 + epsilon)


def get_recall_values(dataset_distances, run_distances, count, threshold, epsilon=1e-3):
    recalls = np.zeros(len(run_distances))
    for i in range(len(run_distances)):
        t = threshold(dataset_distances[i], count, epsilon)
        actual = 0
        for d in run_distances[i][:count]:
            if d <= t:
                actual += 1
        recalls[i] = actual
    return (np.mean(recalls) / float(count), np.std(recalls) / float(count), recalls)


def knn(dataset_distances, run_distances, count, metrics, epsilon=1e-3):
    if "knn" not in metrics:
        knn_metrics = metrics.create_group("knn")
        mean, std, recalls = get_recall_values(dataset_distances, run_distances, count, knn_threshold, epsilon)
        knn_metrics.attrs["mean"] = mean
        knn_metrics.attrs["std"] = std
        knn_metrics["recalls"] = recalls
    else:
        print("Found cached result")
    return metrics["knn"]


def epsilon(dataset_distances, run_distances, count, metrics, epsilon=0.01):
    s = "eps" + str(epsilon)
    if s not in metrics:
        epsilon_metrics = metrics.create_group(s)
        mean, std, recalls = get_recall_values(dataset_distances, run_distances, count, epsilon_threshold, epsilon)
        epsilon_metrics.attrs["mean"] = mean
        epsilon_metrics.attrs["std"] = std
        epsilon_metrics["recalls"] = recalls
    else:
        print("Found cached result")
    return metrics[s]


def rel(dataset_distances, run_distances, metrics):
    if "rel" not in metrics.attrs:
        total_closest_distance = 0.0
        total_candidate_distance = 0.0
        for true_distances, found_distances in zip(dataset_distances, run_distances):
            total_closest_distance += np.sum(true_distances)
            total_candidate_distance += np.sum(found_distances)
        if total_closest_distance < 0.01:
            metrics.attrs["rel"] = float("inf")
        else:
            metrics.attrs["rel"] = total_candidate_distance / total_closest_distance
    else:
        print("Found cached result")
    return metrics.attrs["rel"]


def queries_per_second(queries, attrs):
    return 1.0 / attrs["best_search_time"]


def percentile_50(times):
    return np.percentile(times, 50.0) * 1000.0


def percentile_95(times):
    return np.percentile(times, 95.0) * 1000.0


def percentile_99(times):
    return np.percentile(times, 99.0) * 1000.0


def percentile_999(times):
    return np.percentile(times, 99.9) * 1000.0


def index_size(queries, attrs):
    # TODO(erikbern): should replace this with peak memory usage or something
    return attrs.get("index_size", 0)


def build_time(queries, attrs):
    return attrs["build_time"]


def candidates(queries, attrs):
    return attrs["candidates"]


def dist_computations(queries, attrs):
    return attrs.get("dist_comps", 0) / (attrs["run_count"] * len(queries))

class DummyMetric:
    def __init__(self):
        self.attrs = {}
        self.d = {}

    def __getitem__(self, key):
        return self.d.get(key, None)

    def __setitem__(self, key, value):
        self.d[key] = value

    def __contains__(self, key):
        return key in self.d

    def create_group(self, name):
        self.d[name] = DummyMetric()
        return self.d[name]

all_metrics = {
    "knn": {
        "type":"knn",
        "description": "Recall",
        "function": lambda true_distances, run_distances, metrics, times, distance_count: knn(
            true_distances, run_distances, distance_count, metrics
        ).attrs[
            "mean"
        ],  # noqa
        "worst": float("-inf"),
        "lim": [0.0, 1.03],
    },
    "epsilon": {
        "type":"epsilon",
        "description": "Epsilon 0.01 Recall",
        "function": lambda true_distances, run_distances, metrics, times, distance_count: epsilon(
            true_distances, run_distances, distance_count, metrics
        ).attrs[
            "mean"
        ],  # noqa
        "worst": float("-inf"),
    },
    "largeepsilon": {
        "type":"largeepsilon",
        "description": "Epsilon 0.1 Recall",
        "function": lambda true_distances, run_distances, metrics, times, distance_count: epsilon(
            true_distances, run_distances, distance_count, metrics, 0.1
        ).attrs[
            "mean"
        ],  # noqa
        "worst": float("-inf"),
    },
    "rel": {
        "type":"rel",
        "description": "Relative Error",
        "function": lambda true_distances, run_distances, metrics, times, distance_count: rel(
            true_distances, run_distances, metrics
        ),  # noqa
        "worst": float("inf"),
    },
    """ "qps": {
        "description": "Queries per second (1/s)",
        "function": lambda true_distances, run_distances, metrics, times, run_attrs: queries_per_second(
            true_distances, run_attrs
        ),  # noqa
        "worst": float("-inf"),
    }, """
    "p50": {
        "description": "Percentile 50 (millis)",
        "function": lambda true_distances, run_distances, metrics, times, distance_count: percentile_50(times),  # noqa
        "worst": float("inf"),
    },
    "p95": {
        "description": "Percentile 95 (millis)",
        "function": lambda true_distances, run_distances, metrics, times, distance_count: percentile_95(times),  # noqa
        "worst": float("inf"),
    },
    "p99": {
        "description": "Percentile 99 (millis)",
        "function": lambda true_distances, run_distances, metrics, times, distance_count: percentile_99(times),  # noqa
        "worst": float("inf"),
    },
    "p999": {
        "description": "Percentile 99.9 (millis)",
        "function": lambda true_distances, run_distances, metrics, times, distance_count: percentile_999(times),  # noqa
        "worst": float("inf"),
    },
    # "distcomps": {
    #     "description": "Distance computations",
    #     "function": lambda true_distances, run_distances, metrics, times, run_attrs: dist_computations(
    #         true_distances, run_attrs
    #     ),  # noqa
    #     "worst": float("inf"),
    # },
    # "build": {
    #     "description": "Build time (s)",
    #     "function": lambda true_distances, run_distances, metrics, times, run_attrs: build_time(
    #         true_distances, run_attrs
    #     ),  # noqa
    #     "worst": float("inf"),
    # },
    # "candidates": {
    #     "description": "Candidates generated",
    #     "function": lambda true_distances, run_distances, metrics, times, run_attrs: candidates(
    #         true_distances, run_attrs
    #     ),  # noqa
    #     "worst": float("inf"),
    # },
    # "indexsize": {
    #     "description": "Index size (kB)",
    #     "function": lambda true_distances, run_distances, metrics, times, run_attrs: index_size(
    #         true_distances, run_attrs
    #     ),  # noqa
    #     "worst": float("inf"),
    # },
    # "queriessize": {
    #     "description": "Index size (kB)/Queries per second (s)",
    #     "function": lambda true_distances, run_distances, metrics, times, run_attrs: index_size(
    #         true_distances, run_attrs
    #     )
    #     / queries_per_second(true_distances, run_attrs),  # noqa
    #     "worst": float("inf"),
    # },
}
