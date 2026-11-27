"""WC kinematics probes: python scripts/panda/eval/wc_kinematics.py --help (frozen or LoRA, e/mu)."""

import argparse
import glob
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler
from safetensors.torch import load_file

from pimm.datasets.transform import Compose
from pimm.datasets.transform.spatial import GridSample
from pimm.datasets.utils import collate_fn
from pimm.datasets.wand import first_light
from pimm.engines.hooks.eval.probes import wc as resolution
from pimm.models.builder import build_model
from pimm.models.lora import LoRAAdapter
from pimm.models.utils import offset2batch
from pimm.models.utils.misc import scatter_mean
from pimm.utils.scheduler import OneCycleLR

COORD_SCALE = 18.1
LORA_STEPS = {100: 25000, 1000: 50000, 10000: 50000}
LORA_CHECKS = {100: 250, 1000: 500, 10000: 500}
# Keep the frozen job's mixed event stream and its per-head schedule progression.
SPECIES = ("e", "mu", "pi", "pi0", "gamma")
MASS_MEV = {"e": 0.51099895, "mu": 105.6583755}
TASKS = tuple(dict(name=f"{kind}_{species}", kind=kind, species=(species,))
              for species in ("e", "mu") for kind in ("vertex", "direction", "momentum"))
FROZEN_STEPS = {100: 96000, 1000: 178125, 10000: 178125}
FROZEN_CHECKS = {100: 240, 1000: 1240, 10000: 3562}
SCHEDULE_SAMPLES, WARMUP_SAMPLES = 10_000_000, 10_000
LR, WD, BATCH_VAL = 3e-4, 0.01, 16
LORA_TARGETS = [f"blocks.{stage}.{block}.{module}" for stage, depth in enumerate((3, 6, 12, 6))
                for block in range(depth)
                for module in (("linear",) if stage < 2 else ("attn.qkv", "attn.proj", "mlp.0", "mlp.2"))]
LORA_TARGETS += [f"pools.{i}.proj" for i in range(3)]


class ProbeSites(Dataset):
    def __init__(self, split, events, inputs, data_root):
        self.events, self.inputs, self.handle, self._pid = np.asarray(events), inputs, None, None
        paths = sorted(glob.glob(str(Path(data_root) / "train/config_01/sensor/wc_sensor_*.h5")))
        if not paths:
            raise FileNotFoundError("No training sensor table under --data-root")
        with h5py.File(paths[0], "r") as handle:
            self.pmt = handle["config/sensor_positions"][:].astype(np.float32)
        voxels = np.floor(self.pmt / COORD_SCALE / np.array(0.03)).astype(np.int64)
        self.pmt_key = GridSample.fnv_hash_vec(voxels - voxels.min(0))
        with h5py.File(inputs, "r") as handle:
            if handle.attrs["split"] != split:
                raise ValueError(f"Input file is not the {split} split")
            where = {(a, b): i for i, (a, b) in enumerate(zip(handle["file"][:].tolist(), handle["event"][:].tolist()))}
            self.offsets = handle["offsets"][:]
        self.rows = np.array([where[(a, b)] for a, b in self.events.tolist()])
        self.transform = Compose([
            dict(type="NormalizeCoord", center=[0., 0., 0.], scale=COORD_SCALE),
            dict(type="Update", keys_dict={"index_valid_keys": ["coord", "energy", "time", "hit", "hidden"]}),
            dict(type="GridSample", grid_size=0.03, hash_type="fnv", mode="train", return_grid_coord=True, sum_keys=["energy"]),
            dict(type="LogTransform", min_val=0.01, max_val=1000., log=True, clip=True, keys=["energy"]),
            dict(type="FourierTime", periods_ns=[8.0 * 4.0 ** k for k in range(11)], subtract_min=False),
            dict(type="ToTensor"),
            dict(type="Update", keys_dict={"grid_size": 0.03}),
            dict(type="Collect", keys=["coord", "grid_coord", "energy", "time", "hit", "hidden", "name"],
                 feat_keys=["coord", "energy", "time", "hit", "hidden"]),
        ])

    def __len__(self):
        return len(self.events)

    def __getitem__(self, index):
        if self.handle is None or self._pid != os.getpid():
            self.handle, self._pid = h5py.File(self.inputs, "r"), os.getpid()
        start, end = self.offsets[self.rows[index]], self.offsets[self.rows[index] + 1]
        sensor, pe, time = (self.handle[key][start:end] for key in ("sensor_idx", "PE", "T"))
        order = np.argsort(sensor, kind="stable")
        pmt, starts = np.unique(sensor[order], return_index=True)
        pmt_pe, pmt_time = np.add.reduceat(pe[order], starts), np.minimum.reduceat(time[order], starts)
        sites, site = np.unique(self.pmt_key[pmt], return_inverse=True)
        n = len(sites)
        energy, times = np.zeros(n, np.float32), np.full(n, np.inf, np.float32)
        np.add.at(energy, site, pmt_pe)
        np.minimum.at(times, site, pmt_time)
        order = np.argsort(site, kind="stable")
        count = np.bincount(site, minlength=n)
        rep = pmt[order[np.cumsum(count) - count + np.random.default_rng(index).integers(0, count)]]
        file, event = self.events[index]
        data = dict(coord=self.pmt[rep], energy=energy[:, None], time=(times - first_light(times)).astype(np.float32),
                    hit=np.ones((n, 1), dtype=bool), hidden=np.zeros((n, 1), np.float32),
                    name=f"wc_sensor_{file:04d}/event_{event:03d}")
        self.transform.transforms[1].keys_dict["index_valid_keys"] = ["coord", "energy", "time", "hit", "hidden"]
        return dict(self.transform(data), event=torch.tensor([index]))


def probe_events(pools, split, n):
    rows, events, species = [], [], []
    for s in SPECIES:
        budget = n if s in ("e", "mu") else n // 2
        count = budget if split == "train" else len(pools[f"{split}_{s}_row"])
        if count > len(pools[f"{split}_{s}_row"]):
            raise ValueError(f"Not enough {split} events of species {s}")
        rows.append(pools[f"{split}_{s}_row"][:count])
        events.append(np.stack([pools[f"{split}_{s}_file"][:count], pools[f"{split}_{s}_event"][:count]], 1))
        species.extend([s] * count)
    species = np.array(species)
    mask = np.stack([species == task["species"][0] for task in TASKS], 1)
    return np.concatenate(rows), np.concatenate(events), species, mask, np.zeros_like(mask)


def load_encoder(path, lora=False, random=False):
    export = Path(path)
    if random:
        torch.manual_seed(0)
    config = json.loads((export / "config.json").read_text())["model"]
    if lora:
        encoder = LoRAAdapter(model=config, rank=16, alpha=32, dropout=0., target_modules=LORA_TARGETS, freeze_batchnorm=True)
        target = encoder.model
    else:
        target = encoder = build_model(config)
    if not random:
        missing, extra = target.load_state_dict(load_file(str(export / "model.safetensors")), strict=False)
        if extra or any("lora_" not in key for key in missing):
            raise RuntimeError(f"Encoder checkpoint mismatch: missing={missing}, extra={extra}")
    if not lora:
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
    return encoder.cuda().eval()


def tuned(encoder):
    return {name: parameter.detach().cpu() for name, parameter in encoder.named_parameters() if parameter.requires_grad}


def truth(index, rows, species):
    first = index['primary_offsets'][rows]
    ke = index['primary_ke_MeV'][first].astype(np.float64)
    mass = np.array([MASS_MEV.get(s, np.nan) for s in species])
    direction = index['primary_dir'][first].astype(np.float64)
    return dict(vertex=index['vertex'][rows].astype(np.float32), direction=(direction / np.linalg.norm(direction, axis=1, keepdims=True)).astype(np.float32), log_momentum=np.log(np.sqrt(ke * (ke + 2 * mass)) / 1000.0).astype(np.float32))

class StreamBatches(Sampler):

    def __init__(self, n_events, batch, start, total, seed=0):
        (self.n, self.batch, self.start, self.total, self.seed) = (n_events, batch, start, total, seed)
        self.per_pass = n_events // batch
        self.pass_order = (None, None)

    def global_batch(self, b):
        (p, k) = divmod(b, self.per_pass)
        if self.pass_order[0] != p:
            generator = torch.Generator().manual_seed(self.seed * 100003 + p)
            self.pass_order = (p, torch.randperm(self.n, generator=generator))
        return self.pass_order[1][k * self.batch:(k + 1) * self.batch].numpy()

    def __iter__(self):
        for b in range(self.start, self.total):
            yield self.global_batch(b).tolist()

    def __len__(self):
        return self.total - self.start

def mlp(c_in, c_out, hidden=256):
    return nn.Sequential(nn.Linear(c_in, hidden), nn.ReLU(inplace=True), nn.Dropout(p=0.3), nn.Linear(hidden, c_out))

class AttentivePool(nn.Module):

    def __init__(self, channels, queries=32):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(queries, channels))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.value = nn.Linear(channels, channels // queries * queries, bias=False)
        self.out_channels = self.value.out_features

    def forward(self, f, ev):
        from torch_scatter.composite import scatter_softmax

        with torch.autocast('cuda', enabled=False):
            a = scatter_softmax(f @ self.query.t(), ev.batch, dim=0)
            pooled = torch.stack([x.t() @ y for (x, y) in zip(a.split(ev.sizes), f.split(ev.sizes))])
            w = self.value.weight.view(len(self.query), -1, f.shape[1])
            return torch.einsum('nmc,mdc->nmd', pooled, w).flatten(1)

class Events:

    def __init__(self, coord, batch, n):
        from torch_scatter import scatter

        (self.batch, self.n) = (batch, n)
        self.sizes = torch.bincount(batch, minlength=n).tolist()
        self.centroid = scatter(coord, batch, dim=0, dim_size=n, reduce='mean')
        self.q = coord - self.centroid[batch]
        q = self.q.double()
        cov = scatter(q[:, :, None] * q[:, None, :], batch, dim=0, dim_size=n, reduce='mean')
        trace = cov.diagonal(dim1=1, dim2=2).sum(1)
        self.cov = cov + 1e-09 * trace[:, None, None] * torch.eye(3, dtype=cov.dtype, device=cov.device)
        self.cubic = scatter(q * (q * q).sum(1, keepdim=True), batch, dim=0, dim_size=n, reduce='mean')

    def subset(self, keep, f):
        sites = keep[self.batch]
        out = Events.__new__(Events)
        out.n = int(keep.sum())
        out.batch = (torch.cumsum(keep, 0) - 1)[self.batch[sites]]
        out.sizes = [s for (s, k) in zip(self.sizes, keep.tolist()) if k]
        out.q = self.q[sites]
        for key in ('centroid', 'cov', 'cubic'):
            setattr(out, key, getattr(self, key)[keep])
        return (f[sites], out)

class Moment(nn.Module):

    def __init__(self, g):
        super().__init__()
        (self.g, self.a) = (g, nn.Parameter(torch.zeros(1)))

    def forward(self, f, ev):
        from torch_scatter import scatter

        s = self.g(f).float()
        m = scatter(ev.q * s, ev.batch, dim=0, dim_size=ev.n, reduce='mean').double()
        return torch.linalg.solve(ev.cov, (m + self.a.double() * ev.cubic)[..., None])[..., 0].float()

class Head(nn.Module):

    def __init__(self, kind, arm, channels=1170):
        super().__init__()
        (self.kind, self.arm) = (kind, arm)
        g = nn.Linear(channels, 1) if arm == 'linear' else None
        if kind in ('momentum',):
            self.pool = AttentivePool(channels) if arm == 'attentive' else None
            self.out = g or mlp(self.pool.out_channels if self.pool else channels, 1)
        else:
            assert arm != 'attentive', 'vertex and direction are moment readouts (linear or MLP)'
            self.moment = Moment(g or mlp(channels, 1))

    def forward(self, f, ev):
        if self.kind in ('momentum',):
            pooled = self.pool(f, ev) if self.pool else scatter_mean(f, ev.batch, ev.n)
            return self.out(pooled)[:, 0].float()
        pred = self.moment(f, ev)
        return COORD_SCALE * (pred + ev.centroid) if self.kind == 'vertex' else pred

def summed_loss(kind, pred, target):
    return ((pred - target) ** 2).sum()

def metrics(kind, pred, target):
    if kind == 'vertex':
        m = resolution.vertex_resolution(target['vertex'], pred, target['direction'])
        return dict(m, score=-m['q68_cm'])
    if kind == 'direction':
        m = resolution.angular_resolution(target, pred)
        return dict(m, score=-m['q68_deg'])
    m = resolution.energy_resolution(np.exp(target), np.exp(pred))
    return dict(m, score=-m['res_pct'])

def encode(encoder, batch):
    inputs = {k: v.cuda(non_blocking=True) for (k, v) in batch.items() if torch.is_tensor(v)}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        point = encoder(inputs)
    return (point.feat.float(), point.coord.float(), offset2batch(point.offset), inputs['event'].cpu())

def targets(kind, t, label, tr, rows):
    return torch.from_numpy({'momentum': tr['log_momentum'], 'vertex': tr['vertex'], 'direction': tr['direction']}[kind][rows])

@torch.no_grad()
def evaluate(encoder, heads, loader, mask):
    for head in heads.values():
        head.eval()
    predictions = {name: [] for name in heads}
    for batch in loader:
        features, coordinates, site_event, event = encode(encoder, batch)
        events = Events(coordinates, site_event, len(event))
        for name, head in heads.items():
            keep = torch.from_numpy(mask[event.numpy(), head.task]).cuda()
            if keep.any():
                with torch.autocast('cuda', dtype=torch.float16):
                    predictions[name].append(head(*events.subset(keep, features)).cpu().numpy())
    for head in heads.values():
        head.train()
    return {name: np.concatenate(values) for name, values in predictions.items() if values}

class ValBatches(Sampler):

    def __init__(self, n_events, size):
        self.numbers = list(range(-(-n_events // size)))
        (self.n, self.size) = (n_events, size)

    def __iter__(self):
        for b in self.numbers:
            yield list(range(b * self.size, min((b + 1) * self.size, self.n)))

    def __len__(self):
        return len(self.numbers)


def make_loader(dataset, batch_sampler, workers):
    kwargs = dict(num_workers=workers, collate_fn=collate_fn, pin_memory=True)
    if workers:
        kwargs['prefetch_factor'] = 4
    return DataLoader(dataset, batch_sampler=batch_sampler, **kwargs)


def task_target(kind, data, keep):
    if kind == 'vertex':
        return dict(vertex=data['truth']['vertex'][keep], direction=data['truth']['direction'][keep])
    return data['truth']['direction' if kind == 'direction' else 'log_momentum'][keep]


def train(args):
    args.batch = args.batch or (56 if args.mode == 'frozen' else 16)
    out = Path(args.out)
    (out / 'best').mkdir(parents=True, exist_ok=True)
    pools = np.load(args.pools)
    index = np.load(Path(args.data_root) / 'metadata/event_index.npz')
    partitions = {}
    for split in ('train', 'val'):
        rows, events, species, mask, label = probe_events(pools, split, args.labels)
        if args.mode == 'lora':
            selected = species == args.species
            rows, events, species, mask, label = [v[selected] for v in (rows, events, species, mask, label)]
        if split == 'val' and args.max_val:
            rows, events, species, mask, label = [v[:args.max_val] for v in (rows, events, species, mask, label)]
        partitions[split] = dict(rows=rows, events=events, species=species, mask=mask, label=label,
                                 truth=truth(index, rows, species))
    training, validation = partitions['train'], partitions['val']
    per_pass = len(training['rows']) // args.batch
    if not per_pass:
        raise ValueError('Training pool is smaller than --batch')
    total = args.steps or (LORA_STEPS[args.labels] if args.mode == 'lora' else FROZEN_STEPS[args.labels])
    schedule = SCHEDULE_SAMPLES // args.batch
    warmup = 625 if args.mode == 'lora' else WARMUP_SAMPLES // args.batch
    if total > schedule:
        raise ValueError('Training steps exceed the shared OneCycle schedule')
    check = args.check or (LORA_CHECKS[args.labels] if args.mode == 'lora' else FROZEN_CHECKS[args.labels])
    print(json.dumps(dict(mode=args.mode, batch=args.batch, steps=total, schedule_steps=schedule, warmup_steps=warmup, check=check)), flush=True)
    torch.manual_seed(args.seed)
    encoder = load_encoder(args.encoder, args.mode == 'lora', args.random)
    heads = {}
    for arm in (['mlp'] if args.mode == 'lora' else args.arms):
        for t, task in enumerate(TASKS):
            if args.mode == 'lora' and task['species'][0] != args.species:
                continue
            if arm == 'attentive' and task['kind'] != 'momentum':
                continue
            name = f"{arm}_{task['name']}"
            heads[name] = Head(task['kind'], arm).cuda()
            heads[name].task = t
    optimizers, schedulers, scalers = {}, {}, {}
    if args.mode == 'lora':
        parameters = [p for p in encoder.parameters() if p.requires_grad]
        parameters += [p for head in heads.values() for p in head.parameters()]
        groups = {'shared': parameters}
    else:
        groups = {name: list(head.parameters()) for name, head in heads.items()}
    for name, parameters in groups.items():
        lr = 1e-4 if args.mode == 'lora' else LR
        optimizer = optimizers[name] = torch.optim.AdamW([
            dict(params=[p for p in parameters if p.ndim > 1]),
            dict(params=[p for p in parameters if p.ndim <= 1], weight_decay=0.)], lr=lr, weight_decay=WD)
        schedulers[name] = OneCycleLR(optimizer, max_lr=lr, total_steps=schedule, pct_start=warmup,
                                      anneal_strategy='cos', div_factor=10., final_div_factor=100.)
        scalers[name] = torch.amp.GradScaler('cuda')
    best = {name: dict(score=-np.inf) for name in heads}
    step, last = 0, out / 'last.pth'
    if last.exists():
        state = torch.load(last, map_location='cuda', weights_only=False)
        if args.mode == 'lora':
            encoder.load_state_dict(state['lora'], strict=False)
        for name, head in heads.items():
            head.load_state_dict(state['heads'][name])
        for name in groups:
            for field, objects in [('optimizers', optimizers), ('schedulers', schedulers), ('scalers', scalers)]:
                objects[name].load_state_dict(state[field][name])
        step, best = state['step'], state['best']
    stream = StreamBatches(len(training['rows']), args.batch, step, total, args.seed)
    train_loader = make_loader(ProbeSites('train', training['events'], args.train_inputs, args.data_root), stream, args.workers)
    val_loader = make_loader(ProbeSites('val', validation['events'], args.val_inputs, args.data_root),
                             ValBatches(len(validation['rows']), BATCH_VAL), args.workers)

    def score(current):
        encoder.eval()
        predictions = evaluate(encoder, heads, val_loader, validation['mask'])
        if args.mode == 'lora':
            encoder.train()
        lines = []
        for name, head in heads.items():
            keep = validation['mask'][:, head.task]
            if not keep.any():
                continue
            result = metrics(head.kind, predictions[name], task_target(head.kind, validation, keep))
            if result['score'] > best[name]['score']:
                best[name] = dict(score=result['score'], step=current, metrics=result)
                checkpoint = dict(state_dict=head.state_dict(), step=current, metrics=result,
                                  pred=predictions[name], rows=validation['rows'][keep])
                if args.mode == 'lora':
                    checkpoint['lora'] = tuned(encoder)
                torch.save(checkpoint, out / 'best' / f'{name}.pth')
            lines.append(dict(step=current, head=name, **result))
        with (out / 'metrics.jsonl').open('a') as handle:
            handle.writelines(json.dumps(row) + '\n' for row in lines)
        (out / 'best.json').write_text(json.dumps(best, indent=2))
        print(json.dumps(dict(step=current, heads=lines)), flush=True)

    def save(current):
        state = dict(step=current, best=best, heads={k: v.state_dict() for k, v in heads.items()},
                     optimizers={k: v.state_dict() for k, v in optimizers.items()},
                     schedulers={k: v.state_dict() for k, v in schedulers.items()},
                     scalers={k: v.state_dict() for k, v in scalers.items()})
        if args.mode == 'lora':
            state['lora'] = tuned(encoder)
        torch.save(state, str(last) + '.tmp')
        os.replace(str(last) + '.tmp', last)

    if args.mode == 'lora':
        encoder.train()
    for batch in train_loader:
        if args.mode == 'lora':
            inputs = {k: v.cuda(non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}
            with torch.autocast('cuda', dtype=torch.float16):
                point = encoder(inputs)
            features, coordinates, site_event, event = point.feat.float(), point.coord.float(), offset2batch(point.offset), inputs['event'].cpu()
        else:
            features, coordinates, site_event, event = encode(encoder, batch)
        events = Events(coordinates, site_event, len(event))
        rows = event.numpy()
        losses, active = {}, []
        for name, head in heads.items():
            keep = training['mask'][rows, head.task]
            count = int(keep.sum())
            if not count:
                continue
            with torch.autocast('cuda', dtype=torch.float16):
                prediction = head(*events.subset(torch.from_numpy(keep).cuda(), features))
            target = targets(head.kind, head.task, training['label'], training['truth'], rows[keep]).cuda()
            loss = summed_loss(head.kind, prediction, target) / count
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at step {step}: {name}')
            losses[name] = loss
            active.append(name)
        if args.mode == 'lora':
            if not losses:
                raise RuntimeError('No events for the LoRA tasks')
            scaler = scalers['shared']
            scaler.scale(sum(losses.values())).backward()
            scaler.unscale_(optimizers['shared'])
            torch.nn.utils.clip_grad_norm_(groups['shared'], 1.)
            scale = scaler.get_scale()
            scaler.step(optimizers['shared'])
            scaler.update()
            if scaler.get_scale() >= scale:
                schedulers['shared'].step()
            optimizers['shared'].zero_grad(set_to_none=True)
        else:
            if losses:
                sum(scalers[name].scale(loss) for name, loss in losses.items()).backward()
            for name in heads:
                if name in active:
                    scale = scalers[name].get_scale()
                    scalers[name].step(optimizers[name])
                    scalers[name].update()
                    if scalers[name].get_scale() >= scale:
                        schedulers[name].step()
                    optimizers[name].zero_grad(set_to_none=True)
                else:
                    schedulers[name].step()
        step += 1
        if step % check == 0 or step == total:
            print(json.dumps(dict(step=step, train_loss={name: float(loss.detach()) for name, loss in losses.items()})), flush=True)
            score(step)
        if step % min(check, 500) == 0 or step == total:
            save(step)


def kinematics(table, vertex, direction, momentum):
    ok = np.isfinite(momentum) & np.isfinite(vertex).all(1)
    distance = np.linalg.norm(vertex[ok] - table[['vtx_x_m', 'vtx_y_m', 'vtx_z_m']].values[ok], axis=1) * 100
    unit = direction[ok] / np.linalg.norm(direction[ok], axis=1, keepdims=True)
    angle = np.degrees(np.arccos(np.clip((unit * table[['dir_x', 'dir_y', 'dir_z']].values[ok]).sum(1), -1, 1)))
    relative = (momentum[ok] - table.p_MeV.values[ok]) / table.p_MeV.values[ok]
    return dict(vertex_q68_cm=np.quantile(distance, .68), direction_q68_deg=np.quantile(angle, .68),
                dp_q68_pct=100 * np.quantile(np.abs(relative), .68), sigma_dp_pct=100 * relative.std(),
                bias_dp_pct=100 * relative.mean(), no_answer=1 - ok.mean(), events=len(table))


def score_predictions(truth_path, predictions_path, prefix='', split='test'):
    import pandas as pd
    table = pd.read_parquet(truth_path)
    if 'name' not in table:
        table['name'] = [f'config_01/{split}/wc_sensor_{f:04d}/event_{e:03d}' for f, e in zip(table.file, table.event)]
    table = table[table.contained & (table.dwall_m >= 2)].reset_index(drop=True)
    predictions = table[['name']].merge(pd.read_parquet(predictions_path), on='name', how='left')
    result = {}
    for species in ('mu', 'e'):
        select = (table.species == species).values
        columns = {kind: f'{prefix}{kind}_{species}' for kind in ('vertex', 'direction', 'momentum')}
        if not select.any() or not all(f'{column}/x' in predictions or column in predictions for column in columns.values()):
            continue
        vertex = predictions.loc[select, [f"{columns['vertex']}/{axis}" for axis in 'xyz']].values
        direction = predictions.loc[select, [f"{columns['direction']}/{axis}" for axis in 'xyz']].values
        momentum = 1000 * np.exp(predictions.loc[select, columns['momentum']].values)
        result[species] = kinematics(table[select], vertex, direction, momentum)
    return result


@torch.no_grad()
def predict(args):
    import pandas as pd
    table = pd.read_parquet(args.truth)
    if args.max_events:
        table = table.iloc[:args.max_events]
    events = np.stack([table.file.values, table.event.values], 1)
    dataset = ProbeSites(args.split, events, args.inputs, args.data_root)
    data_loader = make_loader(dataset, ValBatches(len(events), args.batch), args.workers)
    encoder = load_encoder(args.encoder, args.lora, args.random)
    result = dict(name=(table.name.values if 'name' in table else
                  [f'config_01/{args.split}/wc_sensor_{f:04d}/event_{e:03d}' for f, e in events]))
    heads, states = {}, {}
    for checkpoint in sorted((Path(args.run_dir) / 'best').glob('*.pth')):
        arm, task_name = checkpoint.stem.split('_', 1)
        if task_name not in {task['name'] for task in TASKS}:
            continue
        kind = task_name.rsplit('_', 1)[0]
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        head = Head(kind, arm).cuda().eval()
        head.load_state_dict(state['state_dict'])
        heads[checkpoint.stem], states[checkpoint.stem] = head, state
    if not heads:
        raise ValueError('No kinematics head checkpoints found under --run-dir/best')
    predictions = {name: [] for name in heads}

    def read_batches(selected):
        for batch in data_loader:
            features, coordinates, event_id, event = encode(encoder, batch)
            geometry = Events(coordinates, event_id, len(event))
            for name, head in selected.items():
                with torch.autocast('cuda', dtype=torch.float16):
                    value = head(features, geometry)
                predictions[name].append(value.float().cpu().numpy())

    if args.lora:
        for name, head in heads.items():
            encoder.load_state_dict(states[name]['lora'], strict=False)
            encoder.eval()
            read_batches({name: head})
    else:
        read_batches(heads)
    for name, values in predictions.items():
        prediction = np.concatenate(values)
        if heads[name].kind in ('vertex', 'direction'):
            result.update({f'{args.prefix}{name}/{axis}': prediction[:, i] for i, axis in enumerate('xyz')})
        else:
            result[f'{args.prefix}{name}'] = prediction
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result).to_parquet(args.out, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest='command', required=True)
    training = subcommands.add_parser('train', help='fit frozen or LoRA kinematics probes')
    training.add_argument('--encoder', required=True, help='encoder export directory')
    training.add_argument('--random', action='store_true', help='initialize the export architecture without loading weights')
    training.add_argument('--mode', choices=('frozen', 'lora'), default='frozen')
    training.add_argument('--species', choices=('e', 'mu'), default='e', help='species for a shared-LoRA kinematics run')
    training.add_argument('--arms', nargs='+', choices=('linear', 'mlp', 'attentive'), default=['linear', 'mlp', 'attentive'])
    training.add_argument('--labels', type=int, choices=list(FROZEN_STEPS), required=True)
    training.add_argument('--data-root', required=True)
    training.add_argument('--pools', required=True)
    training.add_argument('--train-inputs', required=True)
    training.add_argument('--val-inputs', required=True)
    training.add_argument('--out', required=True)
    training.add_argument('--steps', type=int)
    training.add_argument('--check', type=int)
    training.add_argument('--batch', type=int, help='default: 56 for frozen probes, 16 for LoRA')
    training.add_argument('--workers', type=int, default=4)
    training.add_argument('--seed', type=int, default=0)
    training.add_argument('--max-val', type=int, help='bound validation for a smoke run')
    prediction = subcommands.add_parser('predict', help='run each saved head with its selected adapters')
    prediction.add_argument('--encoder', required=True)
    prediction.add_argument('--random', action='store_true')
    prediction.add_argument('--run-dir', required=True)
    prediction.add_argument('--inputs', required=True)
    prediction.add_argument('--truth', required=True)
    prediction.add_argument('--data-root', required=True)
    prediction.add_argument('--split', choices=('val', 'test'), default='test')
    prediction.add_argument('--lora', action='store_true')
    prediction.add_argument('--prefix', default='')
    prediction.add_argument('--batch', type=int, default=64)
    prediction.add_argument('--workers', type=int, default=4)
    prediction.add_argument('--max-events', type=int)
    prediction.add_argument('--out', required=True)
    scoring = subcommands.add_parser('score', help='score a prediction parquet with the paper containment cut')
    scoring.add_argument('--truth', required=True)
    scoring.add_argument('--predictions', required=True)
    scoring.add_argument('--prefix', default='mlp_')
    scoring.add_argument('--split', choices=('val', 'test'), default='test')
    scoring.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.command == 'train':
        train(args)
    elif args.command == 'predict':
        predict(args)
    else:
        result = score_predictions(args.truth, args.predictions, args.prefix, args.split)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, default=lambda value: value.item()))
        print(json.dumps(result, default=lambda value: value.item()))


if __name__ == '__main__':
    main()
