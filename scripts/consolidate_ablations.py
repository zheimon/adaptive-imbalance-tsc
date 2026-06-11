#!/usr/bin/env python3
"""Consolidate ablation result JSONs into experiments/ablation_results.csv."""
import argparse, json, os, glob
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='experiments/ablation_results')
    parser.add_argument('--output', default='experiments/ablation_results.csv')
    args = parser.parse_args()

    json_files = glob.glob(os.path.join(args.results_dir, '*.json'))
    if not json_files:
        print(f"No JSON files found in {args.results_dir}")
        return

    rows = []
    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)
            cfg = data.get('config', {})
            exp = cfg.get('experiment', {})
            mp = cfg.get('method_params', {})
            tm = data.get('test_metrics', {})
            rows.append({
                'experiment_name': exp.get('name',''),
                'variant': exp.get('name','').split('_')[0],
                'dataset': exp.get('dataset',''),
                'architecture': exp.get('architecture',''),
                'seed': exp.get('seed', 0),
                'update_every': mp.get('update_every', None),
                'eta': mp.get('eta', None),
                'alpha': str(mp.get('alpha', [])),
                'macro_f1': tm.get('macro_f1', None),
                'balanced_accuracy': tm.get('balanced_accuracy', None),
                'g_mean': tm.get('g_mean', None),
                'duration_seconds': data.get('duration_seconds', None),
                'timestamp': data.get('timestamp', ''),
            })
        except Exception as e:
            print(f"Error reading {jf}: {e}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} ablation results to {args.output}")
    if not df.empty:
        print(df.groupby('variant')['macro_f1'].agg(['mean','std','count']).to_string())

if __name__ == '__main__':
    main()
