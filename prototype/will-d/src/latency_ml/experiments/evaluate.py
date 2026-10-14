from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any
import numpy as np
from statistics import mean, median, pstdev
import torch
from latency_ml.config import ProjectConfig
from latency_ml.envs.factory import make_latency_env
from latency_ml.envs.latency_env import HOLD_STEPS_KEY
from latency_ml.seeding import seed_env
from latency_ml.rl.task_policy import TaskPolicy

@dataclass(frozen=True)
class LatencyAggregate:
    latency_ns:int; effective_ns:int; action_steps:int; resolution:str; n_seeds:int; mean_return:float; std_across_seeds:float; mean_survival_s:float; per_seed_returns:list[float]; latency_response:list[dict]=None

def evaluate_policy(policy: TaskPolicy, config: ProjectConfig, latency_ns: int, train_seed: int, episodes: int | None=None) -> dict[str, Any]:
    n=episodes if episodes is not None else int(config.evaluation.control_episodes_per_seed)
    env=make_latency_env(config.task, latency_ns); action_steps=env.action_steps; effective_ns=env.icdi.effective_ns; resolution=env.icdi.resolution.value; env.close()
    returns=[]
    survivals=[]
    failures=0
    truncated=0
    for i in range(n):
        e=make_latency_env(config.task,latency_ns); seed=config.evaluation.episode_seed(train_seed*100+i if hasattr(config.evaluation, "episode_seed") else train_seed+i)
        obs,_=seed_env(e,seed); total=0.0; sim=0; term=False; trunc=False
        while True:
            action=int(policy.action_only(torch.as_tensor(np.asarray(obs),dtype=torch.float32)).item()); obs,r,tv,tr,info=e.step(action); total+=float(r); sim+=int(info.get(HOLD_STEPS_KEY,action_steps)); term=term or tv; trunc=trunc or tr; 
            if tv or tr: break
        returns.append(total); survivals.append(sim/1e9); failures+=int(tv); truncated+=int(trunc); e.close()
    return {"task":config.task,"latency_ns":latency_ns,"effective_ns":effective_ns,"action_steps":action_steps,"resolution":resolution,"seed":train_seed,"mean_return":float(mean(returns)) if returns else 0.0,"median_return":float(median(returns)) if returns else 0.0,"std_return":float(pstdev(returns)) if len(returns)>1 else 0.0,"mean_survival_s":float(mean(survivals)) if survivals else 0.0,"failure_rate":float(failures/len(returns)) if returns else 0.0,"truncation_rate":float(truncated/len(returns)) if returns else 0.0,"episodes":len(returns)}

def aggregate_by_latency(summaries):
    by={}
    for s in summaries: by.setdefault(s["latency_ns"],[]).append(s)
    out=[]
    for lat in sorted(by):
        g=sorted(by[lat], key=lambda s:s["seed"])
        r=tuple(s["mean_return"] for s in g)
        out.append({"latency_ns":lat,"effective_ns":g[0]["effective_ns"],"action_steps":g[0]["action_steps"],"resolution":g[0]["resolution"],"n_seeds":len(g),"mean_return":float(mean(r)),"std_across_seeds":float(pstdev(r)) if len(r)>1 else 0.0,"mean_survival_s":float(mean(s["mean_survival_s"] for s in g)),"per_seed_returns":list(r),"failure_rate":float(mean(s["failure_rate"] for s in g)),"truncation_rate":float(mean(s["truncation_rate"] for s in g))})
    return out

def summary_to_dict(summary): return dict(summary)
def load_policy(config, task, activation=None):
    if isinstance(activation, str):
        from latency_ml.experiments.runner import make_activation
        activation = make_activation(activation)
    from latency_ml.models.architectures import build_control_policy
    return build_control_policy(task, activation=activation)