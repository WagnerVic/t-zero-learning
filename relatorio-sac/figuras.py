import warnings; warnings.filterwarnings("ignore")
import sys, glob; sys.path.insert(0,"relatorio/scripts")
from extrai import ler
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np

R = []
for d in sorted(glob.glob("wandb/run-*")):
    try: n,c,df = ler(d)
    except Exception: continue
    if c.get("env_id")!="Hopper-v5" or c.get("total_timesteps")!=100000 or df.empty: continue
    R.append((d.rsplit("-",1)[-1], c, df))

def sel(**kw):
    out=[]
    for rid,c,df in R:
        s=c.get("sac",{}) or {}
        if kw.get("ppo"):
            if "ppo" in str(c.get("algorithm","")): out.append((rid,c,df))
            continue
        if "ppo" in str(c.get("algorithm","")): continue
        ok=all(s.get(k)==v for k,v in kw.items() if k!="ppo")
        if ok: out.append((rid,c,df))
    return sorted(out, key=lambda x: x[1].get("seed",0))

BASE=dict(updates_per_step=1, tau=0.005, learning_starts=5000, autotune=True, clipped_double_q=True)
def curva(ax, runs, lab, cor, chave="charts/episodic_return_mean_last100", suav=0, x="global_step", div=1000):
    for i,(rid,c,df) in enumerate(runs):
        if chave not in df.columns: continue
        s=df[[x,chave]].dropna().sort_values(x).to_numpy()
        xs,ys=s[:,0]/div, s[:,1]
        if suav and len(ys)>suav: ys=np.convolve(ys,np.ones(suav)/suav,"valid"); xs=xs[suav-1:]
        ax.plot(xs,ys,["-","--",":"][i%3],color=cor,lw=1.5,alpha=.85,
                label=f"{lab} s{c.get('seed')}" if i<3 else None)
def arruma(ax,t,xl,yl):
    ax.set_title(t,fontsize=9.5); ax.grid(alpha=.25); ax.set_xlabel(xl,fontsize=8)
    ax.set_ylabel(yl,fontsize=8); ax.tick_params(labelsize=7); ax.legend(fontsize=6.5,frameon=False)

# F1 Parte 3
fig,axes=plt.subplots(1,2,figsize=(14.4,2.4))
curva(axes[0], sel(**BASE), "inicial", "#1f77b4")
curva(axes[0], sel(**{**BASE,"updates_per_step":2}), "utd=2", "#d62728")
curva(axes[1], sel(**BASE), "inicial", "#1f77b4")
curva(axes[1], sel(**{**BASE,"tau":0.02}), "tau=0.02", "#2ca02c")
curva(axes[1], sel(**{**BASE,"learning_starts":1000}), "ls=1000", "#ff7f0e")
for a,t in zip(axes,["inicial contra updates_per_step=2","inicial contra tau=0.02 e learning_starts=1000"]):
    arruma(a,t,"passos (mil)","retorno (média 100 ep.)")
fig.suptitle("Figura 1 — Parte 3: config inicial e candidatos, 3 seeds cada",fontsize=10.5,y=1.03)
fig.tight_layout(); fig.savefig("relatorio-sac/fig1.png",dpi=170,bbox_inches="tight")

# F2 ablacao 1
fig,axes=plt.subplots(1,3,figsize=(14.4,2.4))
for ax,(k,t,yl) in zip(axes,[("losses/alpha","losses/alpha","alpha"),
        ("losses/entropy","entropia (alvo = -3,0)","entropia"),
        ("charts/episodic_return_mean_last100","retorno","retorno")]):
    curva(ax, sel(**BASE), "autotune", "#1f77b4", k, suav=15 if k!="charts/episodic_return_mean_last100" else 0)
    curva(ax, sel(**{**BASE,"autotune":False}), "alpha fixo 0,09", "#d62728", k, suav=15 if k!="charts/episodic_return_mean_last100" else 0)
    arruma(ax,t,"passos (mil)",yl)
axes[1].axhline(-3.0,color="gray",ls=":",lw=1)
fig.suptitle("Figura 2 — Ablação 1: temperatura fixa",fontsize=10.5,y=1.03)
fig.tight_layout(); fig.savefig("relatorio-sac/fig2.png",dpi=170,bbox_inches="tight")

# F3 ablacao 2
fig,axes=plt.subplots(1,3,figsize=(14.4,2.4))
for ax,(k,t,yl) in zip(axes,[("charts/q_minus_return","q_minus_return (positivo = superestima)","Q - retorno"),
        ("losses/qf1_values","losses/qf1_values","Q1"),
        ("charts/episodic_return_mean_last100","retorno","retorno")]):
    curva(ax, sel(**BASE), "dois críticos", "#1f77b4", k, suav=15 if k!="charts/episodic_return_mean_last100" else 0)
    curva(ax, sel(**{**BASE,"clipped_double_q":False}), "um crítico", "#d62728", k, suav=15 if k!="charts/episodic_return_mean_last100" else 0)
    arruma(ax,t,"passos (mil)",yl)
axes[0].axhline(0,color="gray",ls=":",lw=1)
fig.suptitle("Figura 3 — Ablação 2: um crítico só",fontsize=10.5,y=1.03)
fig.tight_layout(); fig.savefig("relatorio-sac/fig3.png",dpi=170,bbox_inches="tight")

# F4 Parte 5
fig,axes=plt.subplots(1,2,figsize=(14.4,2.4))
curva(axes[0], sel(**BASE), "SAC", "#1f77b4"); curva(axes[0], sel(ppo=True), "PPO", "#d62728")
curva(axes[1], sel(**BASE), "SAC", "#1f77b4", x="_runtime", div=1)
curva(axes[1], sel(ppo=True), "PPO", "#d62728", x="_runtime", div=1)
arruma(axes[0],"mesmo orçamento de PASSOS (100k)","passos (mil)","retorno")
arruma(axes[1],"mesmo eixo de TEMPO de relógio","segundos","retorno")
fig.suptitle("Figura 4 — Parte 5: SAC contra PPO contínuo",fontsize=10.5,y=1.03)
fig.tight_layout(); fig.savefig("relatorio-sac/fig4.png",dpi=170,bbox_inches="tight")
print("4 figuras geradas")
