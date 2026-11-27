"""WC probes: python scripts/panda/eval/wc_probes.py {prepare,extract,frozen,lora,score} --help."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from pimm.models.utils.misc import scatter_mean

try:
    from . import wc_kinematics as kin
except ImportError:
    import wc_kinematics as kin

CLASSES = ('e', 'mu', 'pi', 'pi0')
PAIRS = {'mu/e': ('mu', 'e'), 'mu/pi': ('mu', 'pi'), 'e/pi0': ('e', 'pi0')}
STEPS = {100: 96000, 1000: 178125, 10000: 178125}
CHECK = {100: 240, 1000: 1240, 10000: 3562}
LORA_STEPS = {100: 12500, 1000: 25000, 10000: 25000}
SCHEDULE, WARMUP, BATCH, LR = 178571, 178, 16, 3e-4


def split_features(features, truth, device='cuda'):
    table = pd.read_parquet(truth)
    values = np.load(features)
    keep = (table.contained & (table.dwall_m >= 2) & table.species.isin(CLASSES)).values
    return torch.from_numpy(values[keep].astype(np.float32)).to(device), table.species.values[keep], table.index_row.values[keep]


def make_head(n_out, arm, device='cuda'):
    if arm == 'linear':
        return nn.Linear(1170, n_out).to(device)
    return kin.mlp(1170, n_out).to(device)


def pair_aucs(logits, species, pairs=PAIRS):
    result = {}
    for pair, (signal, background) in pairs.items():
        keep = np.isin(species, (signal, background))
        score = logits[keep, CLASSES.index(signal)] - logits[keep, CLASSES.index(background)]
        result[pair] = float(roc_auc_score(species[keep] == signal, score))
    return result


@torch.no_grad()
def predict(head, features):
    head.eval()
    values = head(features).float().cpu().numpy()
    head.train()
    return values


def train(features, target, n_out, n, validation, pairs, arm, seed=0, steps=None, schedule=SCHEDULE, check=None):
    torch.manual_seed(seed)
    head = make_head(n_out, arm, features.device)
    optimizer = torch.optim.AdamW([
        dict(params=[p for p in head.parameters() if p.ndim > 1], weight_decay=.01),
        dict(params=[p for p in head.parameters() if p.ndim <= 1], weight_decay=0.)], lr=LR)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, total_steps=schedule,
                                                   pct_start=WARMUP / schedule, anneal_strategy='cos',
                                                   div_factor=10, final_div_factor=100)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    permutation, position, best = torch.randperm(len(features), generator=generator), 0, None
    total = steps or STEPS[n]
    for step in range(1, total + 1):
        if position + BATCH > len(permutation):
            permutation, position = torch.randperm(len(features), generator=generator), 0
        selected = permutation[position:position + BATCH].to(features.device)
        position += BATCH
        logits = head(features[selected])
        loss = F.cross_entropy(logits, target[selected])
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % (check or CHECK[n]) == 0 or step == total:
            aucs = pair_aucs(predict(head, validation[0]), validation[1], pairs)
            score = float(np.mean(list(aucs.values())))
            if best is None or score > best['score']:
                best = dict(step=step, score=score, val=aucs,
                            state={key: value.clone() for key, value in head.state_dict().items()})
    head.load_state_dict(best.pop('state'))
    return best, head


def frozen(args):
    bank = np.load(args.train_features)
    selected = bank['rank'] < args.per_class
    features = torch.from_numpy(bank['pooled'][selected].astype(np.float32)).to(args.device)
    target = torch.from_numpy(np.array([CLASSES.index(s) for s in bank['species'][selected]])).to(args.device)
    validation = split_features(args.val_features, args.val_truth, args.device)
    test = split_features(args.test_features, args.test_truth, args.device)
    best, head = train(features, target, len(CLASSES), args.per_class, validation, PAIRS, args.arm,
                       seed=args.seed, steps=args.steps, schedule=args.schedule, check=args.check)
    logits = predict(head, test[0])
    result = dict(best, test=pair_aucs(logits, test[1]), per_class=args.per_class, arm=args.arm)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    torch.save(dict(head=head.state_dict(), **result), out / 'best.pth')  # result already holds arm
    pd.DataFrame(dict(row=test[2], species=test[1], **{f'logit_{c}': logits[:, i] for i, c in enumerate(CLASSES)})).to_parquet(out / 'test_logits.parquet', index=False)
    print(json.dumps(result), flush=True)


def split_events(pools, split, per_class=None, max_per_class=None):
    events, species = [], []
    for name in CLASSES:
        count = per_class if split == 'train' else len(pools[f'{split}_{name}_row'])
        if max_per_class:
            count = min(count, max_per_class)
        if count > len(pools[f'{split}_{name}_row']):
            raise ValueError(f'Not enough {split} events for species {name}')
        events.append(np.stack([pools[f'{split}_{name}_file'][:count], pools[f'{split}_{name}_event'][:count]], 1))
        species.extend([name] * count)
    return np.concatenate(events), np.array(species)


@torch.no_grad()
def predict_lora(encoder, head, loader, n):
    encoder.eval()
    head.eval()
    result = np.zeros((n, len(CLASSES)), np.float32)
    for batch in loader:
        logits, event = head_logits(encoder, head, batch)
        result[event.cpu().numpy()] = logits.cpu().numpy()
    encoder.train()
    head.train()
    return result


def head_logits(encoder, head, batch):
    inputs = {key: value.cuda(non_blocking=True) for key, value in batch.items() if torch.is_tensor(value)}
    with torch.autocast('cuda', dtype=torch.float16):
        point = encoder(inputs)
        pooled = scatter(point.feat.float(), kin.offset2batch(point.offset), dim=0, dim_size=len(inputs['event']), reduce='mean')
        return head(pooled).float(), inputs['event'].reshape(-1)


def lora(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pools = np.load(args.pools)
    training_events, training_species = split_events(pools, 'train', args.per_class)
    validation_events, validation_species = split_events(pools, 'val', max_per_class=args.max_per_split)
    table = pd.read_parquet(args.test_truth)
    table = table[(table.contained & (table.dwall_m >= 2) & table.species.isin(CLASSES)).values]
    if args.max_per_split:
        table = table.groupby('species', sort=False).head(args.max_per_split)
    test_species = table.species.values.astype(str)
    test_events = np.stack([table.file.values, table.event.values], 1)
    targets = torch.from_numpy(np.array([CLASSES.index(s) for s in training_species]))
    total = args.steps or LORA_STEPS[args.per_class]
    schedule = 10_000_000 // BATCH
    check = args.check or max(1, total // 50)
    torch.manual_seed(args.seed)
    encoder = kin.load_encoder(args.encoder, lora=True)
    head = kin.mlp(1170, len(CLASSES)).cuda()
    parameters = [p for p in encoder.parameters() if p.requires_grad] + list(head.parameters())
    optimizer = torch.optim.AdamW([
        dict(params=[p for p in parameters if p.ndim > 1]),
        dict(params=[p for p in parameters if p.ndim <= 1], weight_decay=0.)], lr=1e-4, weight_decay=.01)
    scheduler = kin.OneCycleLR(optimizer, max_lr=1e-4, total_steps=schedule, pct_start=625,
                              anneal_strategy='cos', div_factor=10., final_div_factor=100.)
    scaler = torch.amp.GradScaler('cuda')
    step, best, last = 0, dict(score=-np.inf), out / 'last.pth'
    if last.exists():
        state = torch.load(last, map_location='cuda', weights_only=False)
        encoder.load_state_dict(state['lora'], strict=False)
        head.load_state_dict(state['head'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        step, best = state['step'], state['best']
    train_loader = kin.make_loader(kin.ProbeSites('train', training_events, args.train_inputs, args.data_root),
                                    kin.StreamBatches(len(training_events), BATCH, step, total, args.seed), args.workers)
    val_loader = kin.make_loader(kin.ProbeSites('val', validation_events, args.val_inputs, args.data_root),
                                  kin.ValBatches(len(validation_events), kin.BATCH_VAL), args.workers)
    encoder.train()
    for batch in train_loader:
        logits, event = head_logits(encoder, head, batch)
        target = targets[event.cpu()].cuda()
        loss = F.cross_entropy(logits, target)
        if not torch.isfinite(loss):
            raise RuntimeError(f'Nonfinite loss at step {step}')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= scale:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if step % check == 0 or step == total:
            print(json.dumps(dict(step=step, train_loss=float(loss.detach()))), flush=True)
            aucs = pair_aucs(predict_lora(encoder, head, val_loader, len(validation_events)), validation_species)
            score = float(np.mean(list(aucs.values())))
            if score > best['score']:
                best = dict(score=score, step=step, val=aucs)
                torch.save(dict(head=head.state_dict(), lora=kin.tuned(encoder), step=step, val=aucs), out / 'best.pth')
            with (out / 'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(dict(step=step, score=score, **aucs)) + '\n')
        if step % min(check, 500) == 0 or step == total:
            torch.save(dict(step=step, best=best, lora=kin.tuned(encoder), head=head.state_dict(),
                            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict()), str(last) + '.tmp')
            Path(str(last) + '.tmp').replace(last)
    state = torch.load(out / 'best.pth', map_location='cuda', weights_only=False)
    encoder.load_state_dict(state['lora'], strict=False)
    head.load_state_dict(state['head'])
    test_loader = kin.make_loader(kin.ProbeSites('test', test_events, args.test_inputs, args.data_root),
                                   kin.ValBatches(len(test_events), kin.BATCH_VAL), args.workers)
    logits = predict_lora(encoder, head, test_loader, len(test_events))
    result = dict(per_class=args.per_class, steps=total, best_step=best['step'], val=best['val'], test=pair_aucs(logits, test_species))
    (out / 'result.json').write_text(json.dumps(result, indent=2))
    pd.DataFrame(dict(row=table.index_row.values, species=test_species, **{f'logit_{c}': logits[:, i] for i, c in enumerate(CLASSES)})).to_parquet(out / 'test_logits.parquet', index=False)
    print(json.dumps(result), flush=True)


@torch.no_grad()
def extract(args):
    if args.split == 'train':
        pools = np.load(args.pools)
        events, species = split_events(pools, 'train', args.per_class)
        rank = np.concatenate([np.arange(args.per_class)] * len(CLASSES))
    else:
        table = pd.read_parquet(args.truth)
        events, species = np.stack([table.file.values, table.event.values], 1), table.species.values
    dataset = kin.ProbeSites(args.split, events, args.inputs, args.data_root)
    loader = kin.make_loader(dataset, kin.ValBatches(len(events), args.batch), args.workers)
    torch.manual_seed(args.seed)
    encoder = kin.load_encoder(args.encoder, random=args.random)
    values = []
    for batch in loader:
        features, _, event_id, event = kin.encode(encoder, batch)
        values.append(scatter_mean(features, event_id, len(event)).half().cpu().numpy())
    pooled = np.concatenate(values)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if args.split == 'train':
        np.savez(args.out, pooled=pooled, species=species, rank=rank)
    else:
        np.save(args.out, pooled)


def main():
    if __package__:
        from ._wc_inputs import add_parser
    else:
        from _wc_inputs import add_parser

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    add_parser(commands)
    frozen_parser = commands.add_parser('frozen', help='fit a linear or MLP four-class head on cached features')
    frozen_parser.add_argument('--per-class', type=int, choices=list(STEPS), required=True)
    frozen_parser.add_argument('--arm', choices=('linear', 'mlp'), default='mlp')
    frozen_parser.add_argument('--train-features', required=True, help='NPZ containing pooled, species and per-class rank')
    for split in ('val', 'test'):
        frozen_parser.add_argument(f'--{split}-features', required=True)
        frozen_parser.add_argument(f'--{split}-truth', required=True)
    frozen_parser.add_argument('--steps', type=int)
    frozen_parser.add_argument('--schedule', type=int, default=SCHEDULE)
    frozen_parser.add_argument('--check', type=int)
    frozen_parser.add_argument('--device', default='cuda')
    frozen_parser.add_argument('--seed', type=int, default=0)
    frozen_parser.add_argument('--out', required=True)
    lora_parser = commands.add_parser('lora', help='train a four-class MLP and rank-16 encoder adapters')
    lora_parser.add_argument('--per-class', type=int, choices=list(LORA_STEPS), required=True)
    lora_parser.add_argument('--encoder', required=True, help='encoder export directory')
    lora_parser.add_argument('--data-root', required=True)
    lora_parser.add_argument('--pools', required=True)
    for split in ('train', 'val', 'test'):
        lora_parser.add_argument(f'--{split}-inputs', required=True)
    lora_parser.add_argument('--test-truth', required=True)
    lora_parser.add_argument('--steps', type=int)
    lora_parser.add_argument('--check', type=int)
    lora_parser.add_argument('--workers', type=int, default=4)
    lora_parser.add_argument('--seed', type=int, default=0)
    lora_parser.add_argument('--max-per-split', type=int, help='bound validation and test per species for smoke tests')
    lora_parser.add_argument('--out', required=True)
    extraction = commands.add_parser('extract', help='save frozen mean-pooled encoder features')
    extraction.add_argument('--encoder', required=True)
    extraction.add_argument('--random', action='store_true')
    extraction.add_argument('--data-root', required=True)
    extraction.add_argument('--inputs', required=True)
    extraction.add_argument('--split', choices=('train', 'val', 'test'), required=True)
    extraction.add_argument('--pools', help='pool NPZ for train')
    extraction.add_argument('--truth', help='truth parquet for val/test')
    extraction.add_argument('--per-class', type=int, default=10000)
    extraction.add_argument('--batch', type=int, default=64)
    extraction.add_argument('--workers', type=int, default=4)
    extraction.add_argument('--seed', type=int, default=0)
    extraction.add_argument('--out', required=True)
    scoring = commands.add_parser('score', help='score saved four-class logits')
    scoring.add_argument('--logits', required=True)
    scoring.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.command == 'frozen':
        frozen(args)
    elif args.command == 'lora':
        lora(args)
    elif args.command == 'extract':
        if (args.split == 'train' and not args.pools) or (args.split != 'train' and not args.truth):
            parser.error('extract requires --pools for train and --truth for val/test')
        extract(args)
    elif args.command == 'prepare':
        args.prepare_handler(args)
    else:
        table = pd.read_parquet(args.logits)
        result = pair_aucs(table[[f'logit_{name}' for name in CLASSES]].values, table.species.values)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(json.dumps(result))


if __name__ == '__main__':
    main()
