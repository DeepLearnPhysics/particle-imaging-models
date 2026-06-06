"""Runtime profiling hooks."""

import sys
import os
import torch

from .default import HookBase
from .builder import HOOKS

@HOOKS.register_module()
class RuntimeProfiler(HookBase):
    """Run a short torch profiler trace before normal training proceeds."""

    def __init__(
        self,
        forward=True,
        backward=True,
        interrupt=False,
        warm_up=2,
        sort_by="cuda_time_total",
        row_limit=30,
        memory=True,
    ):
        self.forward = forward
        self.backward = backward
        self.interrupt = interrupt
        self.warm_up = warm_up
        self.sort_by = sort_by
        self.row_limit = row_limit
        self.memory = memory

    def before_train(self):
        self.trainer.logger.info("Profiling runtime ...")
        from torch.profiler import profile, record_function, ProfilerActivity
        if self.memory:
            torch.cuda.memory._record_memory_history()


        logdir = self.trainer.cfg.save_path + "/logdir/"
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        # schedule needs: wait + warmup + active steps (times repeat)
        # loop runs warm_up + 1 iterations, so we match the schedule accordingly
        num_steps = self.warm_up + 1
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                logdir, use_gzip=True
            ),
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=max(1, num_steps - 1), repeat=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for i, input_dict in enumerate(self.trainer.train_loader):
                if i == num_steps:
                    break
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].cuda(non_blocking=True)
                if self.forward:
                    # with record_function("model_forward"):
                    output_dict = self.trainer.model(input_dict)
                else:
                    output_dict = self.trainer.model(input_dict)

                loss = output_dict["loss"]

                if self.backward:
                    # with record_function("model_backward"):
                    loss.backward()
                prof.step()
                
                self.trainer.logger.info(f"Profile: [{i + 1}/{num_steps}]")

        if self.forward or self.backward:
            self.trainer.logger.info(
                "Profile: \n"
                + str(
                    prof.key_averages().table(
                        sort_by=self.sort_by, row_limit=self.row_limit
                    )
                )
            )
            # prof.export_chrome_trace(
            #     os.path.join(self.trainer.cfg.save_path, "trace.json")
            # )

        if self.memory:
            torch.cuda.memory._dump_snapshot(
                os.path.join(self.trainer.cfg.save_path, "memory_snapshot.pickle")
            )
        if self.interrupt:
            sys.exit(0)
