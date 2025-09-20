# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import os
import random
import sys
import torch
import torch.distributed as dist

from rl_games.common.algo_observer import AlgoObserver

from . import pbt_utils
from .mutation import mutate
from .pbt_cfg import PbtCfg

# i.e. value for target objective when it is not known
_UNINITIALIZED_VALUE = float(-1e9)


class PbtAlgoObserver(AlgoObserver):
    """rl_games observer that implements Population-Based Training for a single policy process."""

    def __init__(self, params, args_cli):
        """Initialize observer, print the mutation table, and allocate the restart flag.

        Args:
            params (dict): Full agent/task params (Hydra style).
            args_cli: Parsed CLI args used to reconstruct a restart command.
        """
        super().__init__()
        self.printer = pbt_utils.PbtTablePrinter()
        self.dir = params["pbt"]["directory"]

        self.rendering_args = pbt_utils.RenderingArgs(args_cli)
        self.wandb_args = pbt_utils.WandbArgs(args_cli)
        self.env_args = pbt_utils.EnvArgs(args_cli)
        self.distributed_args = pbt_utils.DistributedArgs(args_cli)
        self.cfg = PbtCfg(**params["pbt"])
        self.pbt_it = -1  # dummy value, stands for "not initialized"
        self.score = _UNINITIALIZED_VALUE
        self.pbt_params = pbt_utils.filter_params(pbt_utils.flatten_dict({"agent": params}), self.cfg.mutation)

        assert len(self.pbt_params) > 0, "[DANGER]: Dictionary that contains params to mutate is empty"
        self.printer.print_params_table(self.pbt_params, header="List of params to mutate")

        self.device = params["params"]["config"]["device"]
        self.restart_flag = torch.tensor([0], device=self.device)

    def after_init(self, algo):
        """Capture training directories on rank 0 and create this policy's workspace folder.

        Args:
            algo: rl_games algorithm object (provides writer, train_dir, frame counter, etc.).
        """
        if self.distributed_args.rank != 0:
            return

        self.algo = algo
        self.root_dir = algo.train_dir
        self.ws_dir = os.path.join(self.root_dir, self.cfg.workspace)
        self.curr_policy_dir = os.path.join(self.ws_dir, f"{self.cfg.policy_idx:03d}")
        os.makedirs(self.curr_policy_dir, exist_ok=True)

    def process_infos(self, infos, done_indices):
        """Extract the scalar objective from environment infos and store in `self.score`.

        Notes:
            Expects the objective to be at `infos[self.cfg.objective]` where self.cfg.objective is dotted address.
        """
        score = infos
        for part in self.cfg.objective.split("."):
            score = score[part]
        self.score = score

    def after_steps(self):
        """Main PBT tick executed every train step.

        Flow:
            1) Non-zero ranks: exit immediately if `restart_flag == 1`, else return.
            2) Rank 0: if `restart_flag == 1`, restart this process with new params.
            3) Rank 0: on PBT cadence boundary (`interval_steps`), save checkpoint,
               load population checkpoints, compute bands, and if this policy is an
               underperformer, select a replacement (random leader or self), mutate
               whitelisted params, set `restart_flag`, broadcast (if distributed),
               and print a mutation diff table.
        """
        if self.distributed_args.distributed:
            dist.broadcast(self.restart_flag, src=0)

        if self.distributed_args.rank != 0:
            if self.restart_flag.cpu().item() == 1:
                os._exit(0)
            return

        elif self.restart_flag.cpu().item() == 1:
            self._restart_with_new_params(self.new_params, self.restart_from_checkpoint)
            return

        # Non-zero can continue
        if self.distributed_args.rank != 0:
            return

        if self.pbt_it == -1:
            self.pbt_it = self.algo.frame // self.cfg.interval_steps
            return

        if self.algo.frame // self.cfg.interval_steps <= self.pbt_it:
            return

        self.pbt_it = self.algo.frame // self.cfg.interval_steps
        frame_left = (self.pbt_it + 1) * self.cfg.interval_steps - self.algo.frame
        print(f"Policy {self.cfg.policy_idx}, frames_left {frame_left}, PBT it {self.pbt_it}")
        try:
            pbt_utils.save_pbt_checkpoint(self.curr_policy_dir, self.score, self.pbt_it, self.algo, self.pbt_params)
            ckpts = pbt_utils.load_pbt_ckpts(self.ws_dir, self.cfg.policy_idx, self.cfg.num_policies, self.pbt_it)
            pbt_utils.cleanup(ckpts, self.curr_policy_dir)
        except Exception as exc:
            print(f"Policy {self.cfg.policy_idx}: Exception {exc} during sanity log!")
            return

        sumry = {i: None if c is None else {k: v for k, v in c.items() if k != "params"} for i, c in ckpts.items()}
        self.printer.print_ckpt_summary(sumry)

        policies = list(range(self.cfg.num_policies))
        target_objectives = [ckpts[p]["true_objective"] if ckpts[p] else _UNINITIALIZED_VALUE for p in policies]
        initialized = [(obj, p) for obj, p in zip(target_objectives, policies) if obj > _UNINITIALIZED_VALUE]
        if not initialized:
            print("No policies initialized; skipping PBT iteration.")
            return
        initialized_objectives, initialized_policies = zip(*initialized)

        # 1) Stats
        mean_obj = float(np.mean(initialized_objectives))
        std_obj = float(np.std(initialized_objectives))
        upper_cut = max(mean_obj + self.cfg.threshold_std * std_obj, mean_obj + self.cfg.threshold_abs)
        lower_cut = min(mean_obj - self.cfg.threshold_std * std_obj, mean_obj - self.cfg.threshold_abs)
        leaders = [p for obj, p in zip(initialized_objectives, initialized_policies) if obj > upper_cut]
        underperformers = [p for obj, p in zip(initialized_objectives, initialized_policies) if obj < lower_cut]

        print(f"mean={mean_obj:.4f}, std={std_obj:.4f}, upper={upper_cut:.4f}, lower={lower_cut:.4f}")
        print(f"Leaders: {leaders} Underperformers: {underperformers}")

        # 3) Only replace if *this* policy is an underperformer
        if self.cfg.policy_idx in underperformers:
            # 4) If there are any leaders, pick one at random; else simply mutate with no replacement
            replacement_policy_candidate = random.choice(leaders) if leaders else self.cfg.policy_idx
            print(f"Replacing policy {self.cfg.policy_idx} with {replacement_policy_candidate}.")

            if self.distributed_args.rank == 0:
                for param, value in self.pbt_params.items():
                    self.algo.writer.add_scalar(f"pbt/{param}", value, self.algo.frame)
                self.algo.writer.add_scalar("pbt/00_best_objective", max(initialized_objectives), self.algo.frame)
                self.algo.writer.flush()

            # Decided to replace the policy weights!
            cur_params = ckpts[replacement_policy_candidate]["params"]
            self.new_params = mutate(cur_params, self.cfg.mutation, self.cfg.mutation_rate, self.cfg.change_range)
            self.restart_from_checkpoint = os.path.abspath(ckpts[replacement_policy_candidate]["checkpoint"])
            self.restart_flag[0] = 1
            self.printer.print_mutation_diff(cur_params, self.new_params)

    def _restart_with_new_params(self, new_params, restart_from_checkpoint):
        """Re-exec the current process with a filtered/augmented CLI to apply new params.

        Notes:
            - Filters out existing Hydra-style overrides that will be replaced,
              and appends `--checkpoint=<path>` and new param overrides.
            - On distributed runs, assigns a fresh master port and forwards
              distributed args to the python.sh launcher.
        """
        cli_args = sys.argv
        print(f"previous command line args: {cli_args}")

        SKIP = ["checkpoint"]
        is_hydra = lambda arg: (  # noqa: E731
            (name := arg.split("=", 1)[0]) not in new_params and not any(k in name for k in SKIP)
        )
        modified_args = [cli_args[0]] + [arg for arg in cli_args[1:] if "=" not in arg or is_hydra(arg)]

        modified_args.append(f"--checkpoint={restart_from_checkpoint}")
        modified_args.extend(self.wandb_args.get_args_list())
        modified_args.extend(self.rendering_args.get_args_list())

        # add all of the new (possibly mutated) parameters
        for param, value in new_params.items():
            modified_args.append(f"{param}={value}")

        self.algo.writer.flush()
        self.algo.writer.close()

        if self.wandb_args.enabled:
            import wandb

            # note setdefault will only affect child process, that mean don't have to worry it env variable
            # propagate beyond restarted child process
            os.environ.setdefault("WANDB_RUN_ID", wandb.run.id)  # continue with the same run id
            os.environ.setdefault("WANDB_RESUME", "allow")  # allow wandb to resume
            os.environ.setdefault("WANDB_INIT_TIMEOUT", "300")  # give wandb init more time to be fault tolerant
            wandb.run.finish()

        # Get the directory of the current file
        thisfile_dir = os.path.dirname(os.path.abspath(__file__))
        isaac_sim_path = os.path.abspath(os.path.join(thisfile_dir, "../../../../../_isaac_sim"))
        command = [f"{isaac_sim_path}/python.sh"]

        if self.distributed_args.distributed:
            self.distributed_args.master_port = str(pbt_utils.find_free_port())
            command.extend(self.distributed_args.get_args_list())
        command += [modified_args[0]]
        command.extend(self.env_args.get_args_list())
        command += modified_args[1:]
        if self.distributed_args.distributed:
            command += ["--distributed"]

        print("Running command:", command, flush=True)
        print("sys.executable = ", sys.executable)
        print(f"Policy {self.cfg.policy_idx}: Restarting self with args {modified_args}", flush=True)

        if self.distributed_args.rank == 0:
            pbt_utils.dump_env_sizes()

            # after any sourcing (or before exec’ing python.sh) prevent kept increasing arg_length:
            for var in ("PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "OMNI_USD_RESOLVER_MDL_BUILTIN_PATHS"):
                val = os.environ.get(var)
                if not val or os.pathsep not in val:
                    continue
                seen = set()
                new_parts = []
                for p in val.split(os.pathsep):
                    if p and p not in seen:
                        seen.add(p)
                        new_parts.append(p)
                os.environ[var] = os.pathsep.join(new_parts)

            os.execv(f"{isaac_sim_path}/python.sh", command)


class MultiObserver(AlgoObserver):
    """Meta-observer that allows the user to add several observers."""

    def __init__(self, observers_):
        super().__init__()
        self.observers = observers_

    def _call_multi(self, method, *args_, **kwargs_):
        for o in self.observers:
            getattr(o, method)(*args_, **kwargs_)

    def before_init(self, base_name, config, experiment_name):
        self._call_multi("before_init", base_name, config, experiment_name)

    def after_init(self, algo):
        self._call_multi("after_init", algo)

    def process_infos(self, infos, done_indices):
        self._call_multi("process_infos", infos, done_indices)

    def after_steps(self):
        self._call_multi("after_steps")

    def after_clear_stats(self):
        self._call_multi("after_clear_stats")

    def after_print_stats(self, frame, epoch_num, total_time):
        self._call_multi("after_print_stats", frame, epoch_num, total_time)
