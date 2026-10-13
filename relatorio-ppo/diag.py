"""Resumo de diagnostico de runs de PPO/A2C, lendo os runs offline do wandb."""
import sys, glob, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "relatorio/scripts")
from extrai import ler
import numpy as np

def resumo(d, rotulo=None):
    name, cfg, df = ler(d)
    if df.empty: return None
    p = cfg.get("ppo") or cfg.get("a2c") or {}
    g = lambda k: df[k].dropna().to_numpy() if k in df.columns else np.array([np.nan])
    ret = g("charts/episodic_return_mean_last100")
    fim = lambda a: np.median(a[int(len(a)*0.8):]) if len(a) > 4 else np.nan
    return {
        "rotulo": rotulo or d.rsplit("-",1)[-1],
        "algo": cfg.get("algorithm"), "seed": cfg.get("seed"),
        "passos": cfg.get("total_timesteps"), "envs": cfg.get("num_envs"),
        "nsteps": p.get("num_steps"), "lr": p.get("learning_rate"),
        "epochs": p.get("update_epochs"), "mb": p.get("num_minibatches"),
        "clip": p.get("clip_coef"), "ent": p.get("ent_coef"), "gae": p.get("gae_lambda"),
        "ret_fim": fim(ret), "ret_max": np.nanmax(ret),
        "kl": fim(g("losses/approx_kl")), "clipfrac": fim(g("losses/clipfrac")),
        "entropia": fim(g("losses/entropy")), "expl_var": fim(g("losses/explained_variance")),
        "updates": np.nanmax(g("charts/num_updates")), "sps": fim(g("charts/SPS")),
    }

if __name__ == "__main__":
    dirs = sys.argv[1:] or sorted(glob.glob("wandb/offline-run-*"))[-2:]
    rs = [r for r in (resumo(d) for d in dirs) if r]
    print(f"{'run':<10}{'seed':>5}{'ret_fim':>9}{'ret_max':>9}{'kl':>10}{'clipfr':>8}{'entrop':>8}{'exp.var':>9}{'upd':>7}{'sps':>7}")
    print("-"*83)
    for r in rs:
        print(f"{r['rotulo'][:9]:<10}{r['seed']:>5}{r['ret_fim']:>9.1f}{r['ret_max']:>9.1f}"
              f"{r['kl']:>10.5f}{r['clipfrac']:>8.3f}{r['entropia']:>8.3f}{r['expl_var']:>9.3f}"
              f"{r['updates']:>7.0f}{r['sps']:>7.0f}")
