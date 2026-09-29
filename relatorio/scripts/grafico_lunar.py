"""Figura do extra: LunarLander-v3."""
import glob, json, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
import sys; sys.path.insert(0, "relatorio/scripts")
from extrai import ler

runs = {}
for d in sorted(glob.glob("wandb/offline-run-*")):
    try: name, cfg, df = ler(d)
    except Exception: continue
    if cfg.get("env_id") != "LunarLander-v3" or df.empty: continue
    a = cfg["a2c"]; tt = cfg["total_timesteps"]
    if tt < 1_000_000: continue          # descarta a sonda de 60k
    runs[(tt, cfg["num_envs"], a["num_steps"])] = df

ROT = {
    (3000000, 16, 32): ("n=32, 16 envs, 3M  (minha escolha inicial)", "#d62728", "-"),
    (6000000, 8, 5):   ("n=5, 8 envs, 6M  (resolve: 215)", "#9467bd", "--"),
    (12000000, 16, 5): ("n=5, 16 envs, 12M  (final: 265)", "#1f77b4", "-"),
}
fig, axes = plt.subplots(1, 3, figsize=(14.4, 2.25))
for ax, (chave, titulo, ylab, suav) in zip(axes, [
        ("charts/episodic_return_mean_last100", "retorno (media 100 ep.)", "retorno", 0),
        ("losses/explained_variance", "explained_variance do critico", "expl. var.", 40),
        ("losses/entropy", "entropy (ln 4 = 1,386 e uniforme)", "entropia", 40)]):
    for k, (lab, cor, ls) in ROT.items():
        if k not in runs: continue
        df = runs[k][["global_step", chave]].dropna().sort_values("global_step")
        x, y = df["global_step"].to_numpy(), df[chave].to_numpy()
        if suav and len(y) > suav:
            y = np.convolve(y, np.ones(suav)/suav, mode="valid"); x = x[suav-1:]
        ax.plot(x, y, ls, label=lab, color=cor, lw=1.5, alpha=0.9)
    ax.set_title(titulo, fontsize=10); ax.grid(alpha=0.25)
    ax.set_xlabel("passos de treino", fontsize=8); ax.set_ylabel(ylab, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_formatter(lambda v, p: f"{v/1e6:.0f}M")
axes[0].axhline(200, color="gray", ls=":", lw=1)
axes[0].text(0.3e6, 212, "resolvido (200)", fontsize=6.5, color="gray")
axes[2].axhline(np.log(4), color="gray", ls=":", lw=1)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.10))
fig.suptitle("Extra — LunarLander-v3", fontsize=11, y=1.02)
fig.tight_layout(); fig.savefig("relatorio/fig_lunar.png", dpi=180, bbox_inches="tight")
print("gerado relatorio/fig_lunar.png com", len(runs), "runs")
