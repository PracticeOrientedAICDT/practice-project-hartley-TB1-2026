"""Train identical observation interfaces with/without latency exposure."""
import argparse, json
from dataclasses import asdict
from pathlib import Path
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from envs.beam_env import BeamEnv, BeamConfig

ARCHS = {
    "tiny": [4, 4],
    "small": [8, 8],
    "paper": [16],
    "medium": [16, 16],
    "large": [64, 64],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["zero", "fixed", "random"], default="random")
    p.add_argument("--arch", choices=list(ARCHS) + ["all"], default="all")
    p.add_argument("--steps", type=int, default=500000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--obs-delay", type=int, default=1)
    p.add_argument("--act-delay", type=float, default=7.56)
    p.add_argument("--out", default="models")
    args = p.parse_args()
    torch.set_num_threads(1)
    c = BeamConfig()
    if args.mode == "zero":
        c.obs_delay_range = (0, 0)
        c.action_delay_range = (0.0, 0.0)
    elif args.mode == "fixed":
        c.obs_delay_range = (args.obs_delay,) * 2
        c.action_delay_range = (args.act_delay,) * 2
    # random defaults: observation 0..2 turns, action 0..8 turns.
    for name in ARCHS if args.arch == "all" else [args.arch]:
        env = make_vec_env(lambda: BeamEnv(c), n_envs=args.envs, seed=args.seed)
        model = PPO(
            "MlpPolicy",
            env,
            device="cpu",
            seed=args.seed,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            policy_kwargs=dict(
                activation_fn=torch.nn.ReLU, net_arch=dict(pi=ARCHS[name], vf=[64, 64])
            ),
        )
        model.learn(total_timesteps=args.steps)
        path = Path(args.out) / f"beam_{name}_{args.mode}_s{args.seed}"
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)
        path.with_suffix(".json").write_text(
            json.dumps(
                dict(
                    config=asdict(c),
                    architecture=ARCHS[name],
                    mode=args.mode,
                    seed=args.seed,
                    actual_timesteps=model.num_timesteps,
                ),
                indent=2,
            )
        )
        env.close()


if __name__ == "__main__":
    main()

# Recommended experiment: train latency-specific policies lat_0ns..lat_5000ns.
