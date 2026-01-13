from train.pipeline import download_data
from core.features import enrich_features
from signals.config import SignalConfig
from signals.labeling import build_triple_barrier_labels
import numpy as np

def main():
    ticker = "TSLA"
    cfg = SignalConfig()  # uses your default horizon/barriers

    df = download_data(ticker)
    df.attrs["ticker"] = ticker
    df = enrich_features(df)

    y, _, _ = build_triple_barrier_labels(df, cfg)

    vals, cnts = np.unique(y, return_counts=True)
    total = cnts.sum()
    dist = {int(v): int(c) for v, c in zip(vals, cnts)}
    pct = {int(v): float(c) / float(total) for v, c in zip(vals, cnts)}

    # Labels: -1 stop-first, 0 timeout, +1 profit-first
    print("Label counts:", dist)
    print("Label pct:   ", {k: round(v * 100, 2) for k, v in pct.items()})
    print(f"Total labeled samples: {total}")

if __name__ == "__main__":
    main()
