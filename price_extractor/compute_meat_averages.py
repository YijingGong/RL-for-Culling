#!/usr/bin/env python3
"""Compute average Breaker/Boner/Cutter prices for weight > 500 lbs

Reads `usda_2957_cull_carcass_prices_MANUL.csv` (assumed in same folder),
filters rows for '500 lbs' (and up), and computes mean price per category
using `price_avg` when available, otherwise midpoint of `price_low`/`price_high`.

Outputs a small CSV `usda_2957_cull_meat_averages.csv` and prints the results.
"""
import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
IN_CSV = os.path.join(BASE_DIR, 'usda_2957_cull_carcass_prices_MANUL.csv')
OUT_CSV = os.path.join(BASE_DIR, 'usda_2957_cull_meat_averages.csv')


def load_and_clean(path):
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip() for c in df.columns]
    # numeric conversions
    for col in ('price_low', 'price_high', 'price_avg'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def compute_averages(df):
    # filter to weight_class containing '500' (covers '500 lbs and up')
    mask_weight = df['weight_class'].str.contains(r'500 lbs and up', case=False, na=False)
    # focus on the three target categories
    mask_cat = df['category'].str.contains(r'Breaker|Boner|Cutter', case=False, na=False)
    df2 = df[mask_weight & mask_cat].copy()

    if df2.empty:
        print('No rows found for 500+ lbs Breaker/Boner/Cutter in', IN_CSV)
        return pd.DataFrame()

    # compute a fallback avg when price_avg missing
    df2['price_mid'] = df2[['price_low', 'price_high']].mean(axis=1)
    # use price_avg when present, else midpoint
    df2['price_effective'] = df2['price_avg'].where(df2['price_avg'].notna(), df2['price_mid'])

    # group by category and compute means
    grouped = df2.groupby('category').agg(
        mean_price_effective=('price_effective', 'mean'),
        mean_price_low=('price_low', 'mean'),
        mean_price_high=('price_high', 'mean'),
        months_count=('date', 'nunique'),
        rows_count=('price_effective', 'count'),
    ).reset_index()

    # round numeric columns for readability
    for col in ('mean_price_effective', 'mean_price_low', 'mean_price_high'):
        if col in grouped.columns:
            grouped[col] = grouped[col].round(2)

    return grouped


def main():
    if not os.path.exists(IN_CSV):
        raise SystemExit(f"Input CSV not found: {IN_CSV}")
    df = load_and_clean(IN_CSV)
    out = compute_averages(df)
    if out.empty:
        return
    out.to_csv(OUT_CSV, index=False)
    print('Saved averages to', OUT_CSV)
    print('\nAverages (500 lbs and up):')
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
