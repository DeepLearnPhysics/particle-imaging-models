# Add a hook

Use a hook for behavior attached to the trainer lifecycle: logging, diagnostics,
evaluation, scheduling, checkpoint side effects, or bounded profiling.

## Minimal hook

```python
from pimm.engines.hooks.builder import HOOKS
from pimm.engines.hooks.default import HookBase
from pimm.utils.comm import is_main_process


@HOOKS.register_module()
class MyScalarHook(HookBase):
    def __init__(self, every=100):
        self.every = int(every)

    def after_step(self):
        step = int(self.trainer.global_step)
        if step % self.every:
            return
        if is_main_process() and self.trainer.writer is not None:
            self.trainer.writer.add_scalar("my/value", 1.0, step)
```

Import a new hook module from `pimm/engines/hooks/__init__.py`, then add it to
the ordered config list.

## Distributed rules

| Action | Rule |
|---|---|
| writer/file/log side effect | guard rank 0 |
| collective save/gather/reduction | every rank must enter |
| rank-0 work other ranks depend on | synchronize deliberately |
| access wrapped model internals | unwrap `.module` when present |

Never put a collective behind {py:func}`~pimm.utils.comm.is_main_process`; other
ranks will deadlock.

## State and resume

Generic checkpoints do not serialize arbitrary hook attributes. Derive counters
from `trainer.global_step` when possible. If state affects training correctness,
make its checkpoint behavior explicit and tested; do not hide it in an
unserialized hook buffer. Forward-hook handles should be installed in
`before_train` and removed in `after_train`.

## Hook order

Methods execute in list order. Document dependencies:

- config-mutating naming hooks must run before writer creation through
  `modify_config`;
- evaluator before checkpoint saver;
- parameter-group mutation before a scheduler that consumes those groups;
- checkpoint loader before behavior that inspects loaded parameters.

## Tests

- each overridden lifecycle method at the expected count/order;
- rank-0 guard and collective behavior;
- resume from a nonzero global step;
- cleanup after normal completion and exception where relevant;
- writer absent/off-rank behavior;
- any owned state across checkpoint/restart;
- config construction through `HOOKS`.

Document whether the hook is a monitor or mutates the experiment. Mutating
hooks belong in the resolved config and method description.
