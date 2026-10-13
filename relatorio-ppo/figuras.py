"""Figuras do relatorio de PPO."""
import warnings; warnings.filterwarnings("ignore")
import sys, glob; sys.path.insert(0,"relatorio/scripts")
from extrai import ler
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np

runs = []
for d in sorted(glob.glob("wandb/offline-run-*")):
    try: n,c,df = ler(d)
    except Exception: continue
    if df.empty or c.get("env_id")!="LunarLander-v3": continue
    runs.append((c, df))

def pega(algo, tt, **kw):
    out=[]
    for c,df in runs:
        if c.get("algorithm")!=algo or c.get("total_timesteps")!=tt: continue
        p=c.get(algo,{}) or {}
        if all((p.get(k)==v if k in p or k not in c else c.get(k)==v) for k,v in kw.items()): out.append((c,df))
    # dedupe por seed, ficando com o run mais recente (a varredura e a validacao
    # podem compartilhar o mesmo config)
    por_seed = {}
    for c, df in out: por_seed[c.get("seed")] = (c, df)
    return [por_seed[s] for s in sorted(por_seed)]

def linha(ax, df, chave, lab, cor, ls="-", x="global_step", suav=0, div=1):
    s = df[[x,chave]].dropna().sort_values(x).to_numpy()
    xs, ys = s[:,0]/div, s[:,1]
    if suav and len(ys)>suav:
        ys = np.convolve(ys, np.ones(suav)/suav, mode="valid"); xs = xs[suav-1:]
    ax.plot(xs, ys, ls, label=lab, color=cor, lw=1.6, alpha=.9)

def arruma(ax, tit, xl, yl, unidade="M"):
    ax.set_title(tit, fontsize=9.5); ax.grid(alpha=.25)
    ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8); ax.tick_params(labelsize=7)

RET="charts/episodic_return_mean_last100"
base = pega("ppo",1000000, learning_rate=2.5e-4, update_epochs=4)
final= pega("ppo",2000000, learning_rate=1e-3, update_epochs=10, gae_lambda=0.95)
a2c  = pega("a2c",2000000)

# ---------- F1: diagnostico ----------
fig,axes = plt.subplots(1,3, figsize=(14.4,2.3))
for i,(ch,tit,yl) in enumerate([(RET,"retorno (média 100 ep.)","retorno"),
        ("losses/approx_kl","approx_kl (escala log)","KL"),("losses/clipfrac","clipfrac","fração")]):
    for c,df in base: linha(axes[i], df, ch, "config inicial", "#d62728", suav=20 if i else 0, div=1e6)
    for c,df in final: linha(axes[i], df, ch, "config final", "#1f77b4", suav=20 if i else 0, div=1e6)
    arruma(axes[i], tit, "passos (M)", yl)
axes[1].set_yscale("log")
h,l = axes[0].get_legend_handles_labels()
visto, hh, ll = set(), [], []          # uma entrada por rotulo, nao por seed
for hi, li in zip(h, l):
    if li not in visto: visto.add(li); hh.append(hi); ll.append(li)
fig.legend(hh, ll, loc="lower center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(.5,-.10))
fig.suptitle("Figura 1 — por que o config inicial não resolve", fontsize=10.5, y=1.03)
fig.tight_layout(); fig.savefig("relatorio-ppo/fig1.png", dpi=180, bbox_inches="tight")

# ---------- F2: caminho ----------
fig,axes = plt.subplots(1,2, figsize=(14.4,2.3))
cam = [(base,"inicial (lr 2.5e-4, 4 épocas)","#d62728"),
       (pega("ppo",1000000,learning_rate=1e-3,update_epochs=4),"+ lr 1e-3","#ff7f0e"),
       (pega("ppo",1000000,learning_rate=1e-3,update_epochs=10),"+ 10 épocas","#2ca02c"),
       (final,"+ 2M passos (final)","#1f77b4")]
for rs,lab,cor in cam:
    if rs: linha(axes[0], rs[0][1], RET, lab, cor, div=1e6)
arruma(axes[0],"o caminho até resolver (seed 1)","passos (M)","retorno")
axes[0].axhline(200,color="gray",ls=":",lw=1)
for i,(c,df) in enumerate(final):
    linha(axes[1], df, RET, f"final, seed {c['seed']}", ["#1f77b4","#2ca02c"][i], "-" if i==0 else "--", div=1e6)
arruma(axes[1],"config final, 2 seeds","passos (M)","retorno")
axes[1].axhline(200,color="gray",ls=":",lw=1)
for ax in axes: ax.legend(fontsize=7, frameon=False)
fig.suptitle("Figura 2 — das três mudanças ao critério atendido", fontsize=10.5, y=1.03)
fig.tight_layout(); fig.savefig("relatorio-ppo/fig2.png", dpi=180, bbox_inches="tight")

# ---------- F3: PPO vs A2C ----------
fig,axes = plt.subplots(1,2, figsize=(14.4,2.3))
for i,(c,df) in enumerate(final):
    linha(axes[0], df, RET, f"PPO seed {c['seed']}", "#1f77b4", "-" if i==0 else "--", div=1e6)
    linha(axes[1], df, RET, f"PPO seed {c['seed']}", "#1f77b4", "-" if i==0 else "--", x="_runtime")
for i,(c,df) in enumerate(a2c):
    linha(axes[0], df, RET, f"A2C seed {c['seed']}", "#d62728", "-" if i==0 else "--", div=1e6)
    linha(axes[1], df, RET, f"A2C seed {c['seed']}", "#d62728", "-" if i==0 else "--", x="_runtime")
# No eixo de tempo entra tambem o A2C de 12M da atividade anterior: ele gasta
# praticamente o mesmo tempo de relogio que o PPO de 2M, e e a unica forma honesta
# de responder "e se o orcamento fosse o mesmo TEMPO?"
a2c12 = pega("a2c", 12000000)
if a2c12:
    linha(axes[1], a2c12[0][1], RET, "A2C 12M (mesmo tempo)", "#2ca02c", x="_runtime")
arruma(axes[0],"mesmo orçamento de PASSOS (2M)","passos de ambiente (M)","retorno")
arruma(axes[1],"mesmo eixo de TEMPO de relógio","segundos","retorno")
for ax in axes:
    ax.axhline(200,color="gray",ls=":",lw=1); ax.legend(fontsize=7, frameon=False)
fig.suptitle("Figura 3 — PPO contra A2C: a resposta muda com o eixo", fontsize=10.5, y=1.03)
fig.tight_layout(); fig.savefig("relatorio-ppo/fig3.png", dpi=180, bbox_inches="tight")
print("3 figuras geradas:", [len(base),len(final),len(a2c)], "runs usados")
