# Dynamical laser-target environment

The original example used a discrete spring-damper model with two control axes,
Gaussian velocity disturbance and a circular target. The v4 environment preserves
that problem structure while making delay a first-class part of the plant.

## State and plant

Each axis is an underdamped second-order oscillator. The environment propagates
its exact continuous-time transition between command timestamps. A delivered
command changes axis velocity with the same sign convention as the original
script: a positive x action steers velocity in the negative x direction. Wind is
an unknown Gaussian velocity impulse once per sample.

Defaults:

```text
sample rate             10 MHz (100 ns/sample)
natural frequency       0.40 MHz
damping ratio           0.02
control impulse gain     0.10
wind standard deviation 0.02
boundary radius         50 normalized units
target radius            2 normalized units
```

These numbers intentionally remain normalized. They are suitable for investigating
latency sensitivity, but they are not calibrated to a particular galvanometer,
fast steering mirror, laser cavity, camera or quadrant detector. Replace them with
measured frequency response, noise and actuator limits before making hardware claims.

## Latency semantics

At sample `t`, the policy issues a 2-D action. Its delivery timestamp is:

```text
t + (nominal inference + transport + per-command jitter) * sample_rate
```

The queue accepts fractional sample timestamps and multiple in-flight commands.
Nominal latency and total nominal latency are policy inputs. Realized jitter is not.
Training can use a fixed budget per model or a known episode-randomized range.
Evaluation cross-sweeps every trained model across every deployed budget.

The default sample rate places the requested threshold region directly in view:
100 ns is one sample, 250 ns is 2.5 samples, and 500 ns is five samples. A cliff is
not forced. Its location depends on natural frequency, damping, authority, noise,
jitter and whether the policy was trained at the deployed delay.

## Renderer

Each synchronized tile contains:

1. x/y centroid oscilloscope;
2. target-plane trajectory with target circle;
3. x versus normalized x-velocity phase plane;
4. timestamped 2-D command pipeline.

Compared policies use the same reset seed and disturbance sequence. Video FPS only
controls playback. The controller still acts at every simulated sample.

## Meaningful stress sweeps

To investigate failure boundaries, sweep one physical dimension at a time:

- `natural_frequency_hz`: faster plant, less phase margin for a fixed latency;
- `damping_ratio`: weaker passive damping increases controller dependence;
- `control_gain`: lower authority increases recovery time;
- `wind_std`: worsens state prediction;
- `inference_jitter_ns`: makes phase compensation uncertain;
- `sample_hz`: changes latency measured in control samples.

Do not tune only `boundary` or `target_radius` to manufacture a cliff. Those mostly
change scoring thresholds rather than plant dynamics.
