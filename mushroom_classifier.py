from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass
class NaiveBayesModel:
    labels: list[str]
    feature_values: dict[str, set[str]]
    label_counts: Counter[str]
    feature_label_value_counts: dict[str, dict[str, Counter[str]]]
    total_samples: int

    def predict_one(self, sample: dict[str, str]) -> str:
        best_label = self.labels[0]
        best_score = -math.inf

        for label in self.labels:
            prior = (self.label_counts[label] + 1) / (
                self.total_samples + len(self.labels)
            )
            score = math.log(prior)

            for feature, values in self.feature_values.items():
                value = sample.get(feature, "__missing__")
                value_count = self.feature_label_value_counts[feature][label][value]
                denominator = self.label_counts[label] + len(values) + 1
                likelihood = (value_count + 1) / denominator
                score += math.log(likelihood)

            if score > best_score:
                best_score = score
                best_label = label

        return best_label

    def predict(self, samples: list[dict[str, str]]) -> list[str]:
        return [self.predict_one(sample) for sample in samples]


def train_naive_bayes(samples: list[dict[str, str]], labels: list[str]) -> NaiveBayesModel:
    if not samples or not labels or len(samples) != len(labels):
        raise ValueError("Samples and labels must be non-empty and have the same length.")

    label_counts: Counter[str] = Counter(labels)
    features = list(samples[0].keys())
    feature_values = {feature: set() for feature in features}
    feature_label_value_counts: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )

    for sample, label in zip(samples, labels):
        for feature, value in sample.items():
            feature_values[feature].add(value)
            feature_label_value_counts[feature][label][value] += 1

    return NaiveBayesModel(
        labels=sorted(label_counts.keys()),
        feature_values=feature_values,
        label_counts=label_counts,
        feature_label_value_counts=feature_label_value_counts,
        total_samples=len(samples),
    )


def accuracy_score(y_true: list[str], y_pred: list[str]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")
    correct = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == predicted)
    return correct / len(y_true) if y_true else 0.0


def split_train_test(
    samples: list[dict[str, str]], labels: list[str], test_size: float, seed: int = 42
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str], list[str]]:
    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)
    test_count = max(1, int(len(indices) * test_size))
    test_indices = set(indices[:test_count])

    x_train, x_test, y_train, y_test = [], [], [], []
    for index, (sample, label) in enumerate(zip(samples, labels)):
        if index in test_indices:
            x_test.append(sample)
            y_test.append(label)
        else:
            x_train.append(sample)
            y_train.append(label)

    return x_train, x_test, y_train, y_test


def load_dataset(path: str, label_column: str = "class") -> tuple[list[dict[str, str]], list[str]]:
    with open(path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise ValueError("Dataset is empty.")
    if label_column not in rows[0]:
        raise ValueError(f"Label column '{label_column}' not found in dataset.")

    labels = [row[label_column] for row in rows]
    samples = [{k: v for k, v in row.items() if k != label_column} for row in rows]
    return samples, labels


def demo_dataset() -> tuple[list[dict[str, str]], list[str]]:
    samples = [
        {"cap_shape": "x", "odor": "a", "gill_size": "b", "spore_print_color": "k"},
        {"cap_shape": "f", "odor": "l", "gill_size": "n", "spore_print_color": "h"},
        {"cap_shape": "x", "odor": "n", "gill_size": "b", "spore_print_color": "k"},
        {"cap_shape": "k", "odor": "f", "gill_size": "n", "spore_print_color": "r"},
        {"cap_shape": "f", "odor": "n", "gill_size": "b", "spore_print_color": "k"},
        {"cap_shape": "k", "odor": "y", "gill_size": "n", "spore_print_color": "r"},
        {"cap_shape": "x", "odor": "a", "gill_size": "b", "spore_print_color": "n"},
        {"cap_shape": "s", "odor": "s", "gill_size": "n", "spore_print_color": "h"},
        {"cap_shape": "x", "odor": "n", "gill_size": "b", "spore_print_color": "n"},
        {"cap_shape": "f", "odor": "m", "gill_size": "n", "spore_print_color": "r"},
    ]
    labels = [
        "edible",
        "poisonous",
        "edible",
        "poisonous",
        "edible",
        "poisonous",
        "edible",
        "poisonous",
        "edible",
        "poisonous",
    ]
    return samples, labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a mushroom edible/poisonous classifier."
    )
    parser.add_argument("--data", help="Path to CSV dataset file.")
    parser.add_argument(
        "--label-column",
        default="class",
        help="Name of the label column in the CSV dataset.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Fraction of dataset to reserve for testing (0 < test_size < 1).",
    )
    args = parser.parse_args()

    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")

    if args.data:
        samples, labels = load_dataset(args.data, args.label_column)
    else:
        samples, labels = demo_dataset()

    x_train, x_test, y_train, y_test = split_train_test(samples, labels, args.test_size)
    model = train_naive_bayes(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Train samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
