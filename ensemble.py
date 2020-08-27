import pandas as pd
import numpy as np
from glob import glob
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sub-dir', type=str, default='./subs')
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    
    subs = [pd.read_csv(csv) for csv in sorted(glob(os.path.join(args.sub_dir, '*csv')))]
    sub_probs = [sub.target.rank(pct=True).values for sub in subs]
    
    wts = [1/18]*18
    assert len(wts)==len(sub_probs)
    sub_ens = np.sum([wts[i]*sub_probs[i] for i in range(len(wts))],axis=0)
    
    df_sub = subs[0]
    df_sub['target'] = sub_ens
    df_sub.to_csv(f"final_sub1.csv",index=False)
    