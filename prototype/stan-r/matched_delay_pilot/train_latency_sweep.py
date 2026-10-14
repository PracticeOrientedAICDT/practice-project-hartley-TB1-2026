"""Train fixed-budget or randomized-budget PPO actors, export and benchmark."""
import argparse, json
from pathlib import Path
from dataclasses import asdict
import numpy as np
from experiment import ARCHS, BUDGETS, benchmark, rollout, host_info
from experiment import make_config, make_env
from actor import NumpyActor


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--environment", choices=["accelerator", "laser"], default="accelerator"
    )
    p.add_argument(
        "--dynamics-profile",
        choices=["deterministic", "mild", "uncertain", "stress"],
        default="deterministic",
    )
    p.add_argument("--budgets-ns", type=float, nargs="+", default=BUDGETS)
    p.add_argument("--arch", choices=list(ARCHS) + ["all"], default="paper16")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--steps", type=int, default=500000)
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--rollout-steps", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=0.0012)
    p.add_argument("--transport-ns", type=float, default=0.0)
    p.add_argument("--inference-jitter-ns", type=float, default=0.0)
    p.add_argument("--obs-delay", type=int, default=0)
    p.add_argument("--random-range-ns", type=float, nargs=2)
    p.add_argument("--benchmark-calls", type=int, default=2000)
    p.add_argument("--out", default="models")
    a = p.parse_args()
    if a.steps < 1 or a.envs < 1 or a.rollout_steps * a.envs < 64:
        p.error("Invalid training size")
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    torch.set_num_threads(1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for arch in ARCHS if a.arch == "all" else [a.arch]:
        for seed in a.seeds:
            for budget in [a.random_range_ns[0]] if a.random_range_ns else a.budgets_ns:
                label = (
                    "random"
                    if a.random_range_ns
                    else f"lat_{budget:g}ns".replace(".", "p")
                )
                profile_tag = (
                    f"_{a.dynamics_profile}" if a.environment == "accelerator" else ""
                )
                stem = out / f"{a.environment}{profile_tag}_{label}_{arch}_s{seed}"
                kwargs = dict(
                    inference_ns=budget,
                    transport_ns=a.transport_ns,
                    inference_range_ns=tuple(a.random_range_ns)
                    if a.random_range_ns
                    else None,
                )
                if a.environment == "accelerator":
                    kwargs["obs_delay_range"] = (a.obs_delay, a.obs_delay)
                else:
                    kwargs["inference_jitter_ns"] = a.inference_jitter_ns
                cfg = make_config(a.environment, **kwargs)
                if a.environment == "accelerator":
                    from envs.accelerator_profiles import apply_profile

                    cfg = apply_profile(cfg, a.dynamics_profile)
                env = make_vec_env(
                    lambda: make_env(a.environment, cfg), n_envs=a.envs, seed=seed
                )
                model = PPO(
                    "MlpPolicy",
                    env,
                    device="cpu",
                    seed=seed,
                    verbose=1,
                    learning_rate=a.learning_rate,
                    n_steps=a.rollout_steps,
                    batch_size=64,
                    n_epochs=a.epochs,
                    gamma=0.99,
                    policy_kwargs=dict(
                        activation_fn=torch.nn.ReLU,
                        net_arch=dict(pi=ARCHS[arch], vf=[64, 64]),
                    ),
                )
                model.learn(total_timesteps=a.steps)
                model.save(stem)
                actor = NumpyActor.from_ppo(model)
                _, _, observations = rollout(
                    lambda obs: model.predict(obs, deterministic=True)[0],
                    cfg,
                    seed + 10000,
                    environment=a.environment,
                )
                for obs in observations[:512]:
                    np.testing.assert_allclose(
                        actor(obs),
                        model.predict(obs, deterministic=True)[0],
                        atol=2e-6,
                        rtol=2e-5,
                    )
                actor.save(stem.with_suffix(".npz"))
                timings = {
                    backend: benchmark(fn, observations, calls=a.benchmark_calls)
                    for backend, fn in [
                        ("numpy", actor),
                        ("sb3", lambda obs: model.predict(obs, deterministic=True)[0]),
                    ]
                }
                meta = dict(
                    name=stem.name,
                    environment=a.environment,
                    dynamics_profile=a.dynamics_profile,
                    config=asdict(cfg),
                    architecture=ARCHS[arch],
                    seed=seed,
                    trained_inference_ns=budget,
                    actual_timesteps=model.num_timesteps,
                    actor_parameters=sum(w.size + b.size for w, b in actor.weights),
                    actor_macs=sum(w.size for w, b in actor.weights),
                    benchmark=timings,
                    host=host_info(),
                )
                stem.with_suffix(".json").write_text(json.dumps(meta, indent=2))
                manifest.append(str(stem))
                env.close()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
