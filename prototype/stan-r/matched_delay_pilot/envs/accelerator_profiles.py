"""Transparent sensitivity profiles; values are hypotheses, not KARA calibration."""
from dataclasses import replace
from .accelerator_env import AcceleratorConfig

PROFILES={
    'deterministic': dict(
        tune_drift_std_per_turn=0.0,tune_drift_reversion=0.0,
        tune_jump_probability=0.0,tune_jump_std=0.0,momentum_process_std=0.0,
        inference_jitter_ns=0.0,latency_outlier_probability=0.0,
        latency_outlier_ns=0.0,actuator_gain_std_fraction=0.0,
    ),
    'mild': dict(
        tune_drift_std_per_turn=1e-5,
        tune_drift_reversion=0.002,
        tune_jump_probability=2e-4,
        tune_jump_std=5e-4,
        momentum_process_std=2e-4,
        inference_jitter_ns=5.0,
        latency_outlier_probability=1e-4,
        latency_outlier_ns=250.0,
        actuator_gain_std_fraction=0.005,
    ),
    'uncertain': dict(
        tune_drift_std_per_turn=5e-5,
        tune_drift_reversion=0.002,
        tune_jump_probability=1e-3,
        tune_jump_std=0.002,
        momentum_process_std=0.002,
        inference_jitter_ns=25.0,
        latency_outlier_probability=1e-3,
        latency_outlier_ns=500.0,
        actuator_gain_std_fraction=0.02,
    ),
    'stress': dict(
        tune_drift_std_per_turn=1e-4,
        tune_drift_reversion=0.001,
        tune_jump_probability=5e-3,
        tune_jump_std=0.005,
        momentum_process_std=0.005,
        inference_jitter_ns=100.0,
        latency_outlier_probability=1e-2,
        latency_outlier_ns=1000.0,
        actuator_gain_std_fraction=0.05,
    ),
}


def apply_profile(config, name):
    if name not in PROFILES: raise ValueError(f'Unknown profile: {name}')
    return replace(config, **PROFILES[name])
