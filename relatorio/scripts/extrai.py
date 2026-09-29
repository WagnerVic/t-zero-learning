"""Le os runs offline do wandb e identifica cada um pela config gravada.

O nome do run no wandb e {env}__{hash dos env_kwargs}__{seed}__{timestamp}, que nao
carrega os hiperparametros varridos. Por isso o rotulo sai da config, nao do nome.
"""
import glob, json, pickle, warnings
warnings.filterwarnings("ignore")
from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2 as pb
import pandas as pd


def ler(run_dir):
    ds = datastore.DataStore()
    ds.open_for_scan(glob.glob(run_dir + "/*.wandb")[0])
    cfg, rows, name = {}, [], None
    while True:
        rec = ds.scan_data()
        if rec is None:
            break
        r = pb.Record()
        r.ParseFromString(rec)
        w = r.WhichOneof("record_type")
        if w == "history":
            rows.append({(it.nested_key[0] if it.nested_key else it.key): json.loads(it.value_json)
                         for it in r.history.item})
        elif w == "run":
            name = r.run.display_name
            for it in r.run.config.update:
                k = it.nested_key[0] if it.nested_key else it.key
                cfg[k] = json.loads(it.value_json)
    return name, cfg, pd.DataFrame(rows)


def rotulo(cfg):
    a = cfg.get("a2c", {}) or {}
    ne, ns = cfg.get("num_envs"), a.get("num_steps")
    ent, base, seed = a.get("ent_coef"), a.get("use_baseline"), cfg.get("seed")
    if base is False:  return f"Q4 sem baseline (seed {seed})"
    if ent == 0:       return f"Q3 ent_coef=0 (seed {seed})"
    if ent == 0.1:     return "Q3 ent_coef=0.1"
    if ns != 5:        return f"Q2 num_steps={ns}"
    if ne != 8:        return f"Q1 num_envs={ne}"
    return "baseline"


def carrega(raiz="."):
    dados = {}
    for d in sorted(glob.glob(f"{raiz}/wandb/offline-run-*")):
        try:
            name, cfg, df = ler(d)
        except Exception as e:
            print("pulei", d, e); continue
        if not cfg or df.empty or cfg.get("total_timesteps") != 500000:
            continue
        dados[rotulo(cfg)] = {"df": df, "cfg": cfg, "dir": d, "run_name": name}
    return dados


if __name__ == "__main__":
    dados = carrega()
    print(f"\n{len(dados)} runs de 500k passos\n")
    print(f"{'rotulo':<28}{'ret.final':>10}{'entropia':>10}{'adv.std':>9}{'expl.var':>10}{'SPS':>8}")
    print("-" * 75)
    for lab in sorted(dados):
        df = dados[lab]["df"]
        g = lambda k: df[k].dropna()
        print(f"{lab:<28}{g('charts/episodic_return_mean_last100').iloc[-1]:>10.1f}"
              f"{g('losses/entropy').iloc[-1]:>10.3f}{g('charts/advantage_std').iloc[-1]:>9.2f}"
              f"{g('losses/explained_variance').iloc[-1]:>10.3f}{g('charts/SPS').iloc[-1]:>8.0f}")
    pickle.dump(dados, open("relatorio/dados.pkl", "wb"))
    print("\nsalvo em relatorio/dados.pkl")
