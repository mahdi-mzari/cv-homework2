import argparse
import json
import os
import random
import time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

SEED = 42
BATCH = 32
OUT = "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS = MobileNet_V3_Small_Weights.DEFAULT


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


# ==================== Data section ====================
def load_pets(split, transform):
    return torchvision.datasets.OxfordIIITPet("data", split=split, transform=transform, download=True)


def get_data():
    test_tf = WEIGHTS.transforms()
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(test_tf.mean, test_tf.std),
    ])
    train_data = load_pets("trainval", train_tf)
    clean_data = load_pets("trainval", test_tf)
    test_data = load_pets("test", test_tf)
    classes = train_data.classes

    labels = np.array(train_data._labels)
    train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=0.2,
                                          stratify=labels, random_state=SEED)
    with open(f"{OUT}/split_files.json", "w") as f:
        json.dump({"seed": SEED, "train_idx": train_idx.tolist(), "val_idx": val_idx.tolist()}, f)

    splits = [labels[train_idx], labels[val_idx], np.array(test_data._labels)]
    counts = [np.bincount(s, minlength=37) for s in splits]
    rows = ["class,train,val,test"]
    rows += [f"{name},{a},{b},{c}" for name, a, b, c in zip(classes, *counts)]
    rows += [f"TOTAL,{len(splits[0])},{len(splits[1])},{len(splits[2])}"]
    with open(f"{OUT}/class_counts.csv", "w") as f:
        f.write("\n".join(rows) + "\n")

    generator = torch.Generator().manual_seed(SEED)
    loaders = {
        "train": DataLoader(Subset(train_data, train_idx), BATCH, shuffle=True, generator=generator),
        "train_eval": DataLoader(Subset(clean_data, train_idx), BATCH),
        "val": DataLoader(Subset(clean_data, val_idx), BATCH),
        "test": DataLoader(test_data, BATCH),
    }
    return loaders, classes, test_data


# ==================== Model section ====================
def make_model():
    model = mobilenet_v3_small(weights=WEIGHTS)
    model.classifier[3] = nn.Linear(1024, 37)
    return model.to(DEVICE)


# ==================== Train section ====================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    loss, preds, labels = 0.0, [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss += nn.functional.cross_entropy(out, y, reduction="sum").item()
        preds += out.argmax(1).tolist()
        labels += y.tolist()
    return loss / len(labels), np.array(preds), np.array(labels)


def train_network(name, loaders, lr, epochs, use_cosine):
    set_seed()
    model = make_model()
    if name == "head":
        for p in model.features.parameters():
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc, best_epoch, history = -1, 0, []
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        if name == "head":
            model.features.eval()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in loaders["train"]:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
        if use_cosine:
            scheduler.step()

        val_loss, val_preds, val_labels = evaluate(model, loaders["val"])
        val_acc = accuracy_score(val_labels, val_preds)
        history.append({"epoch": epoch, "train_loss": total_loss / total, "train_acc": correct / total,
                        "val_loss": val_loss, "val_acc": val_acc})
        print(f"[{name}] epoch {epoch}: train_acc={correct / total:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), f"{OUT}/{name}_best.pth")

    info = {"trainable_params": sum(p.numel() for p in params), "time_s": time.time() - start,
            "lr": lr, "batch_size": BATCH, "epochs": epochs, "best_epoch": best_epoch,
            "optimizer": "Adam + cosine LR schedule" if use_cosine else "Adam",
            "loss": "CrossEntropy", "history": history}
    model.load_state_dict(torch.load(f"{OUT}/{name}_best.pth"))
    return model, info


# ==================== Test section ====================
def plot_confusion(name, cm, classes):
    plt.figure(figsize=(9, 8))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(37), classes, rotation=90, fontsize=7)
    plt.yticks(range(37), classes, fontsize=7)
    for i, j in zip(*np.nonzero(cm)):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=5, color=color)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion matrix (test) - {name}")
    plt.colorbar(shrink=0.8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/cm_{name}.png", dpi=220)
    plt.close()


def plot_examples(name, y, preds, test_data, classes):
    rng = np.random.RandomState(SEED)
    correct = rng.permutation(np.where(y == preds)[0])[:4]
    wrong = rng.permutation(np.where(y != preds)[0])[:4]
    _, axes = plt.subplots(2, 4, figsize=(12, 6.5))
    for row, (indices, tag) in enumerate([(correct, "correct"), (wrong, "wrong")]):
        for ax, i in zip(axes[row], indices):
            ax.imshow(Image.open(test_data._images[i]).convert("RGB"))
            ax.set_title(f"{tag}\nT: {classes[y[i]]}\nP: {classes[preds[i]]}", fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    plt.suptitle(f"Test predictions - {name}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/examples_{name}.png", dpi=120)
    plt.close()


def test_results(name, outputs, classes, test_data, info):
    result = {"approach": name, **info}
    for split, (y, preds) in outputs.items():
        result[f"{split}_acc"] = accuracy_score(y, preds)
    y, preds = outputs["test"]
    result["test_macro_f1"] = f1_score(y, preds, average="macro")

    cm = confusion_matrix(y, preds, labels=range(37))
    plot_confusion(name, cm, classes)
    plot_examples(name, y, preds, test_data, classes)

    errors = cm.copy()
    np.fill_diagonal(errors, 0)
    top = np.argsort(-errors, axis=None, kind="stable")[:5]
    result["top_confusions"] = [f"{classes[i]} -> {classes[j]} ({errors[i, j]})"
                                for i, j in zip(*np.unravel_index(top, errors.shape))]
    print(f"{name}: test_acc={result['test_acc']:.4f} macro_f1={result['test_macro_f1']:.4f}")
    return result


# ==================== Feature extraction section ====================
@torch.no_grad()
def extract_features(model, loader):
    model.eval()
    features, labels = [], []
    for x, y in loader:
        features.append(model.avgpool(model.features(x.to(DEVICE))).flatten(1).cpu())
        labels.append(y)
    return torch.cat(features).numpy(), torch.cat(labels).numpy()


def run_classifiers(loaders, classes, test_data):
    set_seed()
    model = make_model()
    start = time.time()
    data = {s: extract_features(model, loaders[s]) for s in ["train_eval", "val", "test"]}
    extract_time = time.time() - start

    scaler = StandardScaler().fit(data["train_eval"][0])
    X_train, X_val, X_test = [scaler.transform(X) for X, _ in data.values()]
    y_train, y_val, y_test = [y for _, y in data.values()]

    candidates = {
        "logreg": [LogisticRegression(C=c, max_iter=2000) for c in [0.01, 0.1, 1]],
        "linear_svm": [LinearSVC(C=c, max_iter=5000, random_state=SEED) for c in [0.001, 0.01, 0.1]],
        "random_forest": [RandomForestClassifier(n_estimators=n, random_state=SEED, n_jobs=-1)
                          for n in [100, 300]],
    }
    results = []
    for name, models in candidates.items():
        best_acc = -1
        for clf in models:
            start = time.time()
            clf.fit(X_train, y_train)
            fit_time = time.time() - start
            acc = accuracy_score(y_val, clf.predict(X_val))
            if acc > best_acc:
                best_acc, best, best_time = acc, clf, fit_time

        setting = f"n={best.n_estimators}" if name == "random_forest" else f"C={best.C}"
        info = {"trainable_params": 0, "time_s": extract_time + best_time, "hyperparams": setting,
                "scaling": "StandardScaler (fit on train)", "feature_dim": X_train.shape[1]}
        outputs = {"train": (y_train, best.predict(X_train)), "val": (y_val, best.predict(X_val)),
                   "test": (y_test, best.predict(X_test))}
        results.append(test_results(f"features_{name}", outputs, classes, test_data, info))
    return results


# ==================== Charts and tables section ====================
def plot_curves(results):
    _, axes = plt.subplots(2, 2, figsize=(10, 6.4))
    for row, r in zip(axes, results):
        epochs = [h["epoch"] for h in r["history"]]
        for ax, key, label in zip(row, ["loss", "acc"], ["Loss", "Accuracy"]):
            ax.plot(epochs, [h[f"train_{key}"] for h in r["history"]], marker=".", label="train")
            ax.plot(epochs, [h[f"val_{key}"] for h in r["history"]], marker=".", label="validation")
            ax.axvline(r["best_epoch"], color="gray", linestyle="--", label="selected epoch")
            ax.set(title=f"{r['approach']} - {label.lower()}", xlabel="epoch", ylabel=label)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/curves.png", dpi=150)
    plt.close()


def plot_comparison(results):
    names = [r["approach"].replace("features_", "features\n") for r in results]
    x = np.arange(len(results))
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]})
    for k, split in enumerate(["train", "val", "test"]):
        values = [100 * r[f"{split}_acc"] for r in results]
        ax1.bar_label(ax1.bar(x + (k - 1) * 0.27, values, 0.27, label=split), fmt="%.1f", fontsize=7)
    ax1.set_xticks(x, names, fontsize=8)
    ax1.set(title="Accuracy", ylabel="accuracy (%) - axis starts at 70", ylim=(70, 105))
    ax1.legend(fontsize=8, ncol=3, loc="upper center")
    ax2.bar_label(ax2.bar(x, [r["time_s"] for r in results], color="gray"), fmt="%.0f", fontsize=8)
    ax2.set_xticks(x, names, fontsize=7, rotation=45, ha="right")
    ax2.set(title="Computation time", ylabel="seconds")
    plt.tight_layout()
    plt.savefig(f"{OUT}/comparison.png", dpi=150)
    plt.close()


def save_results(results):
    env = {"torch": torch.__version__, "torchvision": torchvision.__version__,
           "sklearn": sklearn.__version__, "numpy": np.__version__, "seed": SEED, "device": DEVICE,
           "deterministic": "cudnn.deterministic=True, use_deterministic_algorithms(warn_only=True)"}
    with open(f"{OUT}/results.json", "w") as f:
        json.dump({"env": env, "results": results}, f, indent=2)

    lines = ["| Approach | Trainable params | Time (s) | Train acc | Val acc | Test acc | Test macro-F1 |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['approach']} | {r['trainable_params']:,} | {r['time_s']:.1f} | {r['train_acc']:.4f} "
                     f"| {r['val_acc']:.4f} | {r['test_acc']:.4f} | {r['test_macro_f1']:.4f} |")
    with open(f"{OUT}/results_table.md", "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


# ==================== Main section ====================
def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--no_cosine", action="store_true")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    OUT = args.out
    os.makedirs(OUT, exist_ok=True)

    set_seed()
    loaders, classes, test_data = get_data()

    results = []
    for name, lr in [("full", 1e-4), ("head", 1e-3)]:
        model, info = train_network(name, loaders, lr, args.epochs, not args.no_cosine)
        outputs = {}
        for split, loader in [("train", "train_eval"), ("val", "val"), ("test", "test")]:
            _, preds, labels = evaluate(model, loaders[loader])
            outputs[split] = (labels, preds)
        results.append(test_results(name, outputs, classes, test_data, info))
    results += run_classifiers(loaders, classes, test_data)

    save_results(results)
    plot_curves(results[:2])
    plot_comparison(results)


if __name__ == "__main__":
    main()
