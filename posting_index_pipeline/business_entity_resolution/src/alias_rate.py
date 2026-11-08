"""Train-label check for Vishal's U6/U7 France restores: true rate of invented-name aliases (no shared name word) and of
filler/edit copies at the S1's own house number, non-blank, by T6 p band (US/India dev + holdout, error_sweep frames)."""
import os
import pandas as pd
df = pd.read_pickle(f"{os.environ['A']}/train.json.train_pairs.pkl")
m = (df.blank == 'nonblank') & df.addr.isin(['same_no_street', 'same_no_other'])
df['pb'] = pd.cut(df.p, [0, 0.3, 0.64, 0.85, 0.95, 1.0001])
print(df[m].groupby(['name', 'pb']).agg(n=('y', 'size'), true_rate=('y', 'mean'), mean_p=('p', 'mean')).round(3).to_string())
