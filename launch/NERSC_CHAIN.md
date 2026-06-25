# NERSC Submitit Requeue Launches

NERSC short-walltime launches use the generic submitit requeue support in
`pimm submit`.
See `launch/SLURM_CHAIN.md`.

Minimal NERSC example:

```bash
pimm submit \
  --site nersc \
  --chain.jobs 4 \
  --run.name e050-tail-chain \
  --resources.time 02:00:00 \
  --train.config panda/pretrain/pretrain-sonata-v1m1-pilarnet-smallmask
```
