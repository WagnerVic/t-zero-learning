"""Resumo dos runs de SAC lendo os arquivos offline do wandb."""
import warnings; warnings.filterwarnings("ignore")
import sys, glob, json; sys.path.insert(0, "relatorio/scripts")
from extrai import ler
import numpy as np

def runs(env="Hopper-v5", tt=100000):
    for d in sorted(glob.glob("wandb/run-*")):
        try: n, c, df = ler(d)
        except Exception: continue
        if c.get("env_id") != env or c.get("total_timesteps") != tt or df.empty: continue
        yield d, c, df

def fim(df, k, frac=0.9):
    if k not in df.columns: return np.nan
    v = df[k].dropna().to_numpy()
    return np.median(v[int(len(v)*frac):]) if len(v) else np.nan

def resumo(c, df):
    s = c.get("sac", {}) or {}
    return dict(seed=c.get("seed"), id=None,
        utd=s.get("updates_per_step"), tau=s.get("tau"), ls=s.get("learning_starts"),
        qlr=s.get("q_lr"), plr=s.get("policy_lr"), auto=s.get("autotune"),
        alpha_cfg=s.get("alpha"), cdq=s.get("clipped_double_q"), bs=s.get("batch_size"),
        ev=df["eval/mean_return"].dropna().iloc[-1] if "eval/mean_return" in df.columns and len(df["eval/mean_return"].dropna()) else np.nan,
        evs=df["eval/mean_return_sampled"].dropna().iloc[-1] if "eval/mean_return_sampled" in df.columns and len(df["eval/mean_return_sampled"].dropna()) else np.nan,
        last100=fim(df, "charts/episodic_return_mean_last100"),
        alpha=fim(df, "losses/alpha"), ent=fim(df, "losses/entropy"),
        qmr=fim(df, "charts/q_minus_return"), q1=fim(df, "losses/qf1_values"),
        sps=fim(df, "charts/SPS"))

if __name__ == "__main__":
    rs = []
    for d, c, df in runs():
        r = resumo(c, df); r["id"] = d.rsplit("-",1)[-1]; rs.append(r)
    print(f"{len(rs)} runs de 100k no Hopper\n")
    print(f"{'id':<10}{'seed':>5}{'utd':>5}{'tau':>7}{'auto':>6}{'cdq':>6}{'eval':>9}{'eval_s':>9}{'last100':>9}{'alpha':>8}{'q-ret':>9}")
    print("-"*86)
    for r in sorted(rs, key=lambda x: (str(x['utd']), str(x['tau']), x['seed'])):
        print(f"{r['id']:<10}{r['seed']:>5}{str(r['utd']):>5}{str(r['tau']):>7}{str(r['auto'])[:1]:>6}{str(r['cdq'])[:1]:>6}"
              f"{r['ev']:>9.0f}{r['evs']:>9.0f}{r['last100']:>9.0f}{r['alpha']:>8.4f}{r['qmr']:>9.1f}")
