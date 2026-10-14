## Latency-aware control pilot

We train separate reinforcement-learning policies at a sweep of inference delays,
then deploy each policy at the delay it was trained for. This matched-delay test
asks whether latency itself reduces tracking precision or stability after the
controller has learned to compensate for it. It avoids treating train/deploy delay
mismatch as an intrinsic control limit.

The pilot supports a beam-centroid laser model and an accelerator oscillator, with
optional accelerator uncertainty profiles. We compare RMS error, target occupancy,
loss and settling across held-out episodes and independent training seeds. Reports
and videos are ordered numerically by latency. Each saved run has a reusable code
and can be reopened or shared through the Streamlit viewer.

PLAN TO add the plasma thing and also add precise measure of breakdown beyond the current investigation

These are simulation sensitivity studies, not calibrated hardware predictions.
A flat matched-delay curve is as informative as a failure boundary. A claimed cliff
needs consistent evidence across training seeds and adequate training convergence.
