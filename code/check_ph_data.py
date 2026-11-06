#!/usr/bin/env python3
import pandas as pd

df = pd.read_csv('./results/ph_results_ALL.csv')
print(f"Total rows: {len(df)}")
print("\nPer dataset/architecture:")
print(df.groupby(['dataset', 'architecture']).size())
print("\nFirst 10 rows:")
print(df.head(10))
