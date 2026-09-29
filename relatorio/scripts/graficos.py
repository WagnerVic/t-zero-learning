"""Gera uma figura por questao a partir de relatorio/dados.pkl."""
import pickle, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

dados = pickle.load(open("relatorio/dados.pkl", "rb"))
X = "global_step"
CORES = {"baseline": "#1f77b4"}
PALETA = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def serie(lab, chave, suavizar=0):
    df = dados[lab]["df"][[X, chave]].dropna().sort_values(X)
    x, y = df[X].to_numpy(), df[chave].to_numpy()
    if suavizar and len(y) > suavizar:
        k = np.ones(suavizar) / suavizar
        y = np.convolve(y, k, mode="valid")
        x = x[suavizar - 1:]
    return x, y


def painel(ax, labels, chave, titulo, ylabel, suavizar=0, logy=False):
    cores = dict(CORES)
    for i, lab in enumerate([l for l in labels if l != "baseline"]):
        cores.setdefault(lab, PALETA[i % len(PALETA)])
    for lab in labels:
        if lab not in dados:
            continue
        x, y = serie(lab, chave, suavizar)
        estilo = "--" if "seed 2" in lab else "-"
        ax.plot(x, y, estilo, label=lab, color=cores[lab], lw=1.6,
                alpha=0.95 if lab == "baseline" else 0.85)
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel("passos de treino", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    if logy:
        ax.set_yscale("symlog")
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_formatter(lambda v, p: f"{v/1000:.0f}k")


FIGS = [
    ("q1", "Q1 — Numero de atores em paralelo (num_envs)",
     ["baseline", "Q1 num_envs=1", "Q1 num_envs=32"],
     [("charts/episodic_return_mean_last100", "retorno (media 100 ep.)", "retorno", 0, False),
      ("losses/policy_loss", "policy_loss (bruto, sem suavizar)", "policy loss", 0, True),
      ("charts/SPS", "SPS (passos por segundo)", "SPS", 20, False)]),
    ("q2", "Q2 — Horizonte do retorno de n passos (num_steps)",
     ["baseline", "Q2 num_steps=1", "Q2 num_steps=32", "Q2 num_steps=128"],
     [("charts/episodic_return_mean_last100", "retorno (media 100 ep.)", "retorno", 0, False),
      ("losses/value_loss", "value_loss", "value loss", 20, True),
      ("losses/explained_variance", "explained_variance", "expl. var.", 20, False)]),
    ("q3", "Q3 — Coeficiente de entropia (ent_coef)",
     ["baseline", "Q3 ent_coef=0 (seed 1)", "Q3 ent_coef=0 (seed 2)", "Q3 ent_coef=0.1"],
     [("charts/episodic_return_mean_last100", "retorno (media 100 ep.)", "retorno", 0, False),
      ("losses/entropy", "entropy (ln 2 = 0,693 e uniforme)", "entropia", 20, False)]),
    ("q4", "Q4 — Ablacao do baseline (use_baseline)",
     ["baseline", "Q4 sem baseline (seed 1)", "Q4 sem baseline (seed 2)"],
     [("charts/episodic_return_mean_last100", "retorno (media 100 ep.)", "retorno", 0, False),
      ("charts/advantage_std", "advantage_std (escala do peso)", "desvio", 20, True),
      ("charts/advantage_mean", "advantage_mean", "media", 20, True),
      ("losses/entropy", "entropy", "entropia", 20, False)]),
]

for nome, titulo, labels, paineis in FIGS:
    n = len(paineis)
    fig, axes = plt.subplots(1, n, figsize=(14.4, 2.25))
    axes = np.atleast_1d(axes)
    for ax, (chave, sub, ylab, suav, logy) in zip(axes, paineis):
        painel(ax, labels, chave, sub, ylab, suav, logy)
    if nome == "q3":
        axes[1].axhline(np.log(2), color="gray", ls=":", lw=1)
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=min(4, len(lbls)),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(titulo, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(f"relatorio/fig_{nome}.png", dpi=180, bbox_inches="tight")
    print("gerado relatorio/fig_" + nome + ".png")
