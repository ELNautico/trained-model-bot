import pandas as pd
from train.pipeline import download_data

# Outlier dates from your CSV
OUTLIER_DATES = [
    "2021-08-23", "2021-08-24", "2021-08-30", "2021-08-31", "2021-09-01",
    "2021-09-02", "2021-09-03", "2021-03-30", "2022-11-18", "2023-09-18"
]

def check_data_for_outliers(ticker="AAPL"):
    df = download_data(ticker)
    df = df.sort_index()
    print(f"Loaded {len(df)} rows for {ticker}")
    for date in OUTLIER_DATES:
        # Find 5 days before and after
        try:
            idx = df.index.get_loc(date)
        except KeyError:
            print(f"Date {date} not found in data!")
            continue
        window = df.iloc[max(0, idx-5):idx+6]
        print(f"\nData around {date}:")
        print(window[[col for col in window.columns if col.lower() in ['open','high','low','close','volume']]])
        # Check for big jumps
        closes = window['Close'].values
        jumps = pd.Series(closes).pct_change() * 100
        print("% Close change:", jumps.round(2).values)

if __name__ == "__main__":
    check_data_for_outliers()
