import math
import os
import random
import shutil
import time
from collections import deque
from os.path import join
from typing import List

import numpy as np
import copy 

import yaml
from isaaclab_rl.rl_games_utils import flatten_dict
from rl_games.algos_torch.torch_ext import safe_save, safe_filesystem_op
from rl_games.common.algo_observer import AlgoObserver

from .mutation import mutate

import datetime

import torch
import torch.distributed as dist



# i.e. value for target objective when it is not known
_UNINITIALIZED_VALUE = float(-1e9)


def _checkpnt_name(iteration):
    return f'{iteration:06d}.yaml'


def _model_checkpnt_name(iteration):
    return f'{iteration:06d}.pth'


class PbtAlgoObserver(AlgoObserver):
    def __init__(self, params):
        super().__init__()

        self.hydra_dir = params["hydra"]["run"]["dir"]

        self.task_name = params["args_cli"]["task"]       
        self.cli_task_name = params["args_cli"]["task"]
        self.task_num_envs = params["args_cli"]["num_envs"]
        self.camera_enabled = params["args_cli"]["enable_cameras"]
        self.video = params["args_cli"]["video"]
        self.video_length = params["args_cli"]["video_length"]
        self.video_interval = params["args_cli"]["video_interval"]
        
        self.distributed = params["args_cli"]["distributed"]
        self.num_gpus = params["args_cli"]["num_gpus"]
        self.global_rank = params["args_cli"]["global_rank"]    
        
        pbt_params = params['pbt']
        self.pbt_replace_fraction = pbt_params['replace_fraction']
        self.pbt_replace_threshold_frac_std = pbt_params['replace_threshold_frac_std']
        self.pbt_replace_threshold_frac_absolute = pbt_params['replace_threshold_frac_absolute']
        self.pbt_mutation_rate = pbt_params['mutation_rate']
        self.pbt_change_min = pbt_params['change_min']
        self.pbt_change_max = pbt_params['change_max']


        self.dbg_mode = pbt_params['dbg_mode']

        self.policy_idx = pbt_params['policy_idx']
        self.pbt_num_policies = pbt_params['num_policies']
        self.algo = None
        self.pbt_workspace = pbt_params['workspace']
        self.pbt_workspace_dir = self.curr_policy_workspace_dir = None

        self.pbt_iteration = -1  # dummy value, stands for "not initialized"
        self.pbt_interval_steps = pbt_params['interval_steps']
        self.pbt_start_after_steps = pbt_params['start_after']

        self.initial_env_frames = -1  # env frames at the beginning of the experiment, can be > 0 if we resume

        self.pbt_episodes_to_avg = 4096
        self.last_target_objectives = deque([], maxlen=4096)

        self.curr_target_objective_value = _UNINITIALIZED_VALUE
        self.target_objective_known = False  # switch to true when we have enough data to calculate target objective

        self.experiment_start = time.time()
        self.with_wandb = params['wandb_activate']

        # TODO: Fix this 
        # Ensure "agent" key exists
        # Need to add agent because when we are restarting, IsaacLab expects params in agent.params.config.blah
        params_ = copy.deepcopy(params)
        if "agent" not in params_:
            params_["agent"] = {}

        # List the keys to be moved (assuming you want to move all keys except "agent")
        keys_to_move = [key for key in params_ if key != "agent"]

        # Move all selected keys under params["agent"]
        for key in keys_to_move:
            params_["agent"][key] = params_.pop(key)

        self.params = self._flatten_params(params_)
        self.params_to_mutate = pbt_params['mutation']

        self.params = self._filter_params(self.params, self.params_to_mutate)
        assert self.params, "[DANGER]: Dictionary that contains params to mutate is empty"
        print(f'----------------------------------------')
        print(f'List of params to mutate: {self.params=}')

        print(f'{self.with_wandb=}')
        
        self.device = params["params"]["config"]["device"]        
        self.restart_flag = torch.tensor([0], device=self.device)        
        self.count = 0 

    @staticmethod
    def _flatten_params(params, prefix='', separator='.'):
        all_params = flatten_dict(params, prefix, separator)
        return all_params

    @staticmethod
    def _filter_params(params, params_to_mutate):
        filtered_params = dict()

        for key, value in params.items():    

            if key in params_to_mutate:
                if isinstance(value, str):
                    try:
                        # trying to convert values such as "1e-4" to floats because yaml fails to recognize them as such
                        float_value = float(value)
                        value = float_value
                    except ValueError:
                        pass
                
                filtered_params[key] = value
        return filtered_params

    def after_init(self, algo):
        
        if self.global_rank != 0:
            return
        
        self.algo = algo

        self.pbt_workspace_dir = join(algo.train_dir, self.pbt_workspace)
        self.curr_policy_workspace_dir = self._policy_workspace_dir(self.policy_idx)
        os.makedirs(self.curr_policy_workspace_dir, exist_ok=True)

    def process_infos(self, infos, done_indices):
        infos['true_objective'] = infos['episode']['Curriculum/adr']
        if self.global_rank != 0:
            return
        
        if 'true_objective' in infos:
            for done_idx in done_indices:
                true_objective_value = infos['true_objective']
                self.last_target_objectives.append(true_objective_value)

            self.target_objective_known = len(self.last_target_objectives) >= self.last_target_objectives.maxlen
            if self.target_objective_known:
                self.curr_target_objective_value = self.last_target_objectives
        else:
            # environment does not specify "true objective", use regular reward
            self.target_objective_known = self.algo.game_rewards.current_size >= self.algo.games_to_track
            if self.target_objective_known:
                self.curr_target_objective_value = float(self.algo.mean_rewards)

    def _targ_objective_value(self):
        if isinstance(self.curr_target_objective_value, float):
            return self.curr_target_objective_value
        else:
            return float(np.mean(self.curr_target_objective_value))

    def after_steps(self):
                        
        # print(f'{self.restart_flag=}, {self.count=}')    
        if self.distributed:
            dist.broadcast(self.restart_flag, src=0) 

        if self.global_rank != 0:
            self._restart_with_new_params_dryrun()
            return
        
        elif self.global_rank == 0 and self.restart_flag.cpu().item() == 1:
            
            print(f'Restarting the process with new params on {self.global_rank=}, {self.device=}')
            self._restart_with_new_params(self.restart_params['new_params'], 
                                          self.restart_params['restart_from_checkpoint'])
            return 
        
                              
        if self.pbt_iteration == -1:
            self.pbt_iteration = self.algo.frame // self.pbt_interval_steps
            self.initial_env_frames = self.algo.frame
            print(f'Policy {self.policy_idx}: PBT init. Env frames: {self.algo.frame}, pbt_iteration: {self.pbt_iteration}')

        env_frames = self.algo.frame
        iteration = env_frames // self.pbt_interval_steps
        print(f'Policy {self.policy_idx}: Env frames {env_frames/1e6:.1f}M, \
            iteration {iteration}, PBT iteration {self.pbt_iteration}')

        if iteration <= self.pbt_iteration:
            return

        if not self.target_objective_known:
            # not enough data yet to calcuate avg true_objective
            print(f'Policy {self.policy_idx}: Not enough episodes finished, wait for more data...')
            return

        sec_since_experiment_start = time.time() - self.experiment_start
        pbt_start_after_sec = 1 if self.dbg_mode else 30
        if sec_since_experiment_start < pbt_start_after_sec:
            print(f'Policy {self.policy_idx}: Not enough time passed since experiment start {sec_since_experiment_start}')
            return

        if env_frames - self.initial_env_frames < self.pbt_start_after_steps:
            # print(f'Policy {self.policy_idx}: Not enough experience collected to start PBT {env_frames} 
                #   {self.initial_env_frames} {self.pbt_start_after_steps}')
            print(
                f"Policy {self.policy_idx}: Not enough experience to start PBT {env_frames/1e6:.1f} M frames  "
                f"collected {self.initial_env_frames/1e6:.1f} M frames, "
                f"need {self.pbt_start_after_steps/1e6:.1f} M frames."
            )
            return

        print(f'Policy {self.policy_idx}: New pbt iteration {iteration}!')
        self.pbt_iteration = iteration

        try:
            self._save_pbt_checkpoint()
        except Exception as exc:
            print(f'Policy {self.policy_idx}: Exception {exc} when saving PBT checkpoint!')
            return

        try:
            checkpoints = self._load_population_checkpoints()
        except Exception as exc:
            print(f'Policy {self.policy_idx}: Exception {exc} when loading checkpoints!')
            return

        try:
            self._cleanup(checkpoints)
        except Exception as exc:
            print(f'Policy {self.policy_idx}: Exception {exc} during cleanup!')

        print(f'Current policy {self.policy_idx} and {checkpoints=}')

        policies = list(range(self.pbt_num_policies))
        target_objectives = []
        for p in policies:
            if checkpoints[p] is None:
                target_objectives.append(_UNINITIALIZED_VALUE)
            else:
                target_objectives.append(checkpoints[p]['true_objective'])

        policies_sorted = sorted(zip(target_objectives, policies), reverse=True)
        objectives = [objective for objective, p in policies_sorted]
        best_objective = objectives[0]
        policies_sorted = [p for objective, p in policies_sorted]
        best_policy = policies_sorted[0]

        print(f'Policy {self.policy_idx}:  target_objectives={target_objectives}, policy_idx_objective={target_objectives[self.policy_idx]}')

        self._maybe_save_best_policy(best_objective, best_policy, checkpoints[best_policy])

        objectives_filtered = [o for o in objectives if o > _UNINITIALIZED_VALUE]

        try:
            self._pbt_summaries(self.params, best_objective)
        except Exception as exc:
            print(f'Policy {self.policy_idx}: Exception {exc} when writing summaries!')
            return

        replace_fraction = self.pbt_replace_fraction
        replace_number = math.ceil(replace_fraction * self.pbt_num_policies)

        best_policies, best_objectives = policies_sorted[:replace_number], objectives[:replace_number]
        worst_policies, worst_objectives = policies_sorted[replace_number:], objectives[replace_number:]

        print(f'Policy {self.policy_idx}: PBT {best_policies=}, {worst_policies=}')
        print(f'Policy {self.policy_idx}: PBT {best_objectives=}, {worst_objectives=}')

        if self.policy_idx not in worst_policies and not self.dbg_mode:
            # don't touch the policies that are doing okay
            print(f'Current policy {self.policy_idx} is doing well, not among the {worst_policies=}')
            return

        if len(objectives_filtered) <= max(2, self.pbt_num_policies // 2) and not self.dbg_mode:
            print(f'Policy {self.policy_idx}: Not enough data to start PBT, {objectives_filtered}')
            return

        print(f'Current policy {self.policy_idx} is among the {worst_policies=}, consider replacing weights')
        print(f'Policy {self.policy_idx} objective: {self._targ_objective_value()}, {best_objective=} ({best_policy=}).')

        replacement_policy_candidate = random.choice(best_policies)
        candidate_objective = checkpoints[replacement_policy_candidate]['true_objective']
        targ_objective_value = self._targ_objective_value()
        objective_delta = candidate_objective - targ_objective_value

        num_outliers = int(math.floor(0.2 * len(objectives_filtered)))
        print(f'Policy {self.policy_idx} num outliers: {num_outliers}')
        
        print(f'Policy {self.policy_idx} is going to be replaced by {replacement_policy_candidate}')

        if len(objectives_filtered) > num_outliers:
            objectives_filtered_sorted = sorted(objectives_filtered)

            # remove worst policies from the std calculation, this will allow us to keep improving even if 1-2 policies
            # crashed and can't keep improving. Otherwise, std value will be too large.
            objectives_std = np.std(objectives_filtered_sorted[num_outliers:])
        else:
            objectives_std = np.std(objectives_filtered)

        objective_threshold = self.pbt_replace_threshold_frac_std * objectives_std

        absolute_threshold = self.pbt_replace_threshold_frac_absolute * abs(candidate_objective)

        if objective_delta > objective_threshold and objective_delta > absolute_threshold:
            # replace this policy with a candidate
            replacement_policy = replacement_policy_candidate
            print(f'Replacing underperforming policy {self.policy_idx} with {replacement_policy}')
        else:
            print(f'Policy {self.policy_idx}: Difference in objective value ({candidate_objective} vs {targ_objective_value}) is not sufficient to justify replacement,'
                  f'{objective_delta=}, {objectives_std=}, {objective_threshold=}, {absolute_threshold=}')

            # replacing with "self": keep the weights but mutate the hyperparameters
            replacement_policy = self.policy_idx

        # Decided to replace the policy weights!
        new_params = checkpoints[replacement_policy]['params']
        new_params = mutate(new_params, self.params_to_mutate, self.pbt_mutation_rate, self.pbt_change_min, self.pbt_change_max)

        restart_from_checkpoint = os.path.abspath(checkpoints[replacement_policy]['checkpoint'])
        experiment_name = checkpoints[self.policy_idx]['experiment_name']

        try:
            self._pbt_summaries(new_params, best_objective)
        except Exception as exc:
            print(f'Policy {self.policy_idx}: Exception {exc} when writing summaries!')
            return

        print(f'Policy {self.policy_idx}: Preparing to restart the process with mutated parameters!')
        
        self.restart_flag[0] = 1
                
        self.restart_params = dict()
        self.restart_params['new_params'] = new_params
        self.restart_params['restart_from_checkpoint'] = restart_from_checkpoint
        self.restart_params['experiment_name'] = experiment_name
               
        # self._restart_with_new_params(new_params, restart_from_checkpoint, experiment_name)

    def _save_pbt_checkpoint(self):
        if self.global_rank != 0:
            return
        
        """Save PBT-specific information including iteration number, policy index and hyperparameters."""
        checkpoint_file = join(self.curr_policy_workspace_dir, _model_checkpnt_name(self.pbt_iteration))
        
        # It is in 
        # https://github.com/Denys88/rl_games/blob/b483bd62982f668e3fb4d457b418e56fae38ebf2/rl_games/common/a2c_common.py#L630
        
        algo_state = self.algo.get_full_state_weights()
        safe_save(algo_state, checkpoint_file)

        pbt_checkpoint_file = join(self.curr_policy_workspace_dir, _checkpnt_name(self.pbt_iteration))

        pbt_checkpoint = {
            'iteration': self.pbt_iteration,
            'true_objective': self._targ_objective_value(),
            'frame': self.algo.frame,
            'params': self.params,
            'checkpoint': os.path.abspath(checkpoint_file),
            'pbt_checkpoint': os.path.abspath(pbt_checkpoint_file),
            'experiment_name': self.algo.experiment_name,
        }

        print(f'Policy {self.policy_idx}: PBT checkpoint saving the dict {pbt_checkpoint} in {pbt_checkpoint_file} ...')

        with open(pbt_checkpoint_file, 'w') as fobj:
            print(f'Policy {self.policy_idx}: Saving {pbt_checkpoint_file}...')
            yaml.dump(pbt_checkpoint, fobj)

    def _policy_workspace_dir(self, policy_idx):
        return join(self.pbt_workspace_dir, f'{policy_idx:03d}')

    def _load_population_checkpoints(self):
        if self.global_rank != 0:
            return
        
        """
        Load checkpoints for other policies in the population.
        Pick the newest checkpoint, but not newer than our current iteration.
        """
        checkpoints = dict()

        for policy_idx in range(self.pbt_num_policies):
            checkpoints[policy_idx] = None

            policy_workspace_dir = self._policy_workspace_dir(policy_idx)

            if not os.path.isdir(policy_workspace_dir):
                
                print(f'Policy {self.policy_idx}: Not loading {policy_idx} in {policy_workspace_dir} \
                    because it does not exist')
                
                continue

            pbt_checkpoint_files = [f for f in os.listdir(policy_workspace_dir) if f.endswith('.yaml')]
            pbt_checkpoint_files.sort(reverse=True)

            for pbt_checkpoint_file in pbt_checkpoint_files:
                iteration_str = pbt_checkpoint_file.split('.')[0]
                iteration = int(iteration_str)
                
                # current local time
                now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                ctime_ts = os.path.getctime(join(policy_workspace_dir, pbt_checkpoint_file))
                created_str = datetime.datetime.fromtimestamp(ctime_ts).strftime("%Y-%m-%d %H:%M:%S")   

                if iteration <= self.pbt_iteration:
                    with open(join(policy_workspace_dir, pbt_checkpoint_file), 'r') as fobj:
                        print(f'Policy {self.policy_idx} [{now_str}]: Loading policy-{policy_idx} {pbt_checkpoint_file} (created at {created_str})')
                        checkpoints[policy_idx] = safe_filesystem_op(yaml.load, fobj, Loader=yaml.FullLoader)
                        break
                else:
                    print(f'Policy {self.policy_idx}: Not loading {pbt_checkpoint_file} \
                        because current {iteration} < {self.pbt_iteration}')                    
                    pass

        assert self.policy_idx in checkpoints.keys()
        return checkpoints

    def _maybe_save_best_policy(self, best_objective, best_policy_idx: int, best_policy_checkpoint):
        if self.global_rank != 0:
            return
        
        # make a directory containing best policy checkpoints using safe_filesystem_op
        best_policy_workspace_dir = join(self.pbt_workspace_dir, f'best{self.policy_idx}')
        safe_filesystem_op(os.makedirs, best_policy_workspace_dir, exist_ok=True)

        best_objective_so_far = _UNINITIALIZED_VALUE

        best_policy_checkpoint_files = [f for f in os.listdir(best_policy_workspace_dir) if f.endswith('.yaml')]
        best_policy_checkpoint_files.sort(reverse=True)
        if best_policy_checkpoint_files:
            with open(join(best_policy_workspace_dir, best_policy_checkpoint_files[0]), 'r') as fobj:
                best_policy_checkpoint_so_far = safe_filesystem_op(yaml.load, fobj, Loader=yaml.FullLoader)
                best_objective_so_far = best_policy_checkpoint_so_far['true_objective']

        if best_objective_so_far >= best_objective:
            # don't save the checkpoint if it is worse than the best checkpoint so far
            return

        print(f'Policy {self.policy_idx}: New best objective: {best_objective}!')

        # save the best policy checkpoint to this folder
        best_policy_checkpoint_name = f'{self.task_name}_best_obj_{best_objective:015.5f}_iter_{self.pbt_iteration:04d}_policy{best_policy_idx:03d}_frame{self.algo.frame}'

        # copy the checkpoint file to the best policy directory
        try:
            shutil.copy(best_policy_checkpoint['checkpoint'], join(best_policy_workspace_dir, f'{best_policy_checkpoint_name}.pth'))
            shutil.copy(best_policy_checkpoint['pbt_checkpoint'], join(best_policy_workspace_dir, f'{best_policy_checkpoint_name}.yaml'))

            # cleanup older best policy checkpoints, we want to keep only N latest files
            best_policy_checkpoint_files = [f for f in os.listdir(best_policy_workspace_dir)]
            best_policy_checkpoint_files.sort(reverse=True)

            n_to_keep = 6
            for best_policy_checkpoint_file in best_policy_checkpoint_files[n_to_keep:]:
                os.remove(join(best_policy_workspace_dir, best_policy_checkpoint_file))

        except Exception as exc:
            print(f'Policy {self.policy_idx}: Exception {exc} when copying best checkpoint!')
            # no big deal if this fails, hopefully the next time we will succeeed
            return

    def _pbt_summaries(self, params, best_objective):
        if self.global_rank != 0:
            return
        
        for param, value in params.items():
            self.algo.writer.add_scalar(f'zz_pbt/{param}', value, self.algo.frame)
        self.algo.writer.add_scalar(f'zz_pbt/00_best_objective', best_objective, self.algo.frame)
        self.algo.writer.flush()

    def _cleanup(self, checkpoints):
        if self.global_rank != 0:
            return
        
        iterations = []
        for policy_idx, checkpoint in checkpoints.items():
            if checkpoint is None:
                iterations.append(0)
            else:
                iterations.append(checkpoint['iteration'])

        oldest_iteration = sorted(iterations)[0]
        cleanup_threshold = oldest_iteration - 20
        print(f'Policy {self.policy_idx}: Oldest iteration in population is {oldest_iteration}, removing checkpoints older than {cleanup_threshold} iteration')

        pbt_checkpoint_files = [f for f in os.listdir(self.curr_policy_workspace_dir)]

        for f in pbt_checkpoint_files:
            if '.' in f:
                iteration_idx = int(f.split('.')[0])
                if iteration_idx <= cleanup_threshold:
                    print(f'Policy {self.policy_idx}: PBT cleanup: removing checkpoint {f}')
                    # we catch all exceptions in this function so no need to use safe_filesystem_op
                    os.remove(join(self.curr_policy_workspace_dir, f))

        # Sometimes, one of the PBT processes can get stuck, or crash, or be scheduled significantly later on Slurm
        # or a similar cluster management system.
        # In that case, we will accumulate a lot of older checkpoints. In order to keep the number of older checkpoints
        # under control (to avoid running out of disk space) we implement the following logic:
        # when we have more than N checkpoints, we delete half of the oldest checkpoints. This caps the max amount of
        # disk space used, and still allows older policies to participate in PBT

        max_old_checkpoints = 50
        while True:
            pbt_checkpoint_files = [f for f in os.listdir(self.curr_policy_workspace_dir) if f.endswith('.yaml')]
            if len(pbt_checkpoint_files) <= max_old_checkpoints:
                break
            if not self._delete_old_checkpoint(pbt_checkpoint_files):
                break

    def _delete_old_checkpoint(self, pbt_checkpoint_files: List[str]) -> bool:
        if self.global_rank != 0:
            return
        
        """
        Delete the checkpoint that results in the smallest max gap between the remaining checkpoints.
        Do not delete any of the last N checkpoints.
        """
        pbt_checkpoint_files.sort()
        n_latest_to_keep = 20
        candidates = pbt_checkpoint_files[:-n_latest_to_keep]
        num_candidates = len(candidates)
        if num_candidates < 3:
            return False

        def _iter(f):
            return int(f.split('.')[0])

        best_gap = 1e9
        best_candidate = 1
        for i in range(1, num_candidates - 1):
            prev_iteration = _iter(candidates[i - 1])
            next_iteration = _iter(candidates[i + 1])

            # gap is we delete the ith candidate
            gap = next_iteration - prev_iteration
            if gap < best_gap:
                best_gap = gap
                best_candidate = i

        # delete the best candidate
        best_candidate_file = candidates[best_candidate]
        files_to_remove = [best_candidate_file, _model_checkpnt_name(_iter(best_candidate_file))]
        for file_to_remove in files_to_remove:
            print(f'Policy {self.policy_idx}: PBT cleanup old checkpoints, removing checkpoint {file_to_remove} (best gap {best_gap})')
            os.remove(join(self.curr_policy_workspace_dir, file_to_remove))

        return True

    def _restart_with_new_params_dryrun(self):
        
        import os
        reset_item = self.restart_flag.cpu().item()

        if reset_item == 1:
            print('Exiting this process on device = {}'.format(self.device))
            os._exit(0)
        else:
            return

    def _restart_with_new_params(self, new_params, restart_from_checkpoint):
        
        
        import os
        import sys
        import signal
        import socket
        import psutil
                            
        cli_args = sys.argv
        
        print(f'previous command line args: {cli_args}')

        SKIP_KEYS = ['checkpoint', 'hydra.run.dir']

        modified_args = [cli_args[0]]  # initialize with path to the Python script        
        
        for arg in cli_args[1:]:
            if '=' not in arg:
                modified_args.append(arg)
            else:
                assert '=' in arg
                arg_name, arg_value = arg.split('=')
                if arg_name in new_params or any(k in arg_name for k in SKIP_KEYS):
                    continue

                modified_args.append(f'{arg_name}={arg_value}')
        modified_args.append(f'--checkpoint={restart_from_checkpoint}')
        modified_args.append(f'agent.hydra.run.dir={self.hydra_dir}')
        if self.with_wandb:
            import wandb
            modified_args.append('--track')
            modified_args.append(f'--wandb-entity={wandb.run.entity}')
            modified_args.append(f'--wandb-name={wandb.run.name}')
        if self.video:
            modified_args.append('--video')
            modified_args.append(f'--video_length={self.video_length}')
            modified_args.append(f'--video_length={self.video_interval}')
        if self.camera_enabled:
            modified_args.append('--enable_cameras')

        # add all of the new (possibly mutated) parameters
        for param, value in new_params.items():
            modified_args.append(f'{param}={value}')

        self.algo.writer.flush()
        self.algo.writer.close()

        if self.with_wandb:
            import wandb
            wandb.run.finish()



        def _find_free_port(max_tries: int = 20) -> int:
            """
            Return an OS-allocated free TCP port.
            Retries a few times to avoid rare 'bad file descriptor' races.
            """
            for _ in range(max_tries):
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    # Let the kernel pick an available port.
                    s.bind(("", 0))
                    port = s.getsockname()[1]
                    s.close()
                    return port
                except OSError:          # includes 'Bad file descriptor'
                    s.close()
                    continue
            # Fallback: choose a random high port (still extremely unlikely to collide)
            return random.randint(20000, 65000)

        isaac_sim_path = '/workspace/isaaclab/_isaac_sim'
                
        # ---------------------------------------------------------------------
        # Build the torch.distributed command
        # ---------------------------------------------------------------------
        command = [f'{isaac_sim_path}/python.sh'] 

        if self.distributed:
            master_port = _find_free_port()          # new port each restart
            os.environ['MASTER_PORT'] = str(master_port)

            command += [
                '-m', 'torch.distributed.run',
                f'--nnodes=1',
                f'--nproc_per_node={self.num_gpus}',
                f'--master_port={master_port}',
            ]

        command += [
            modified_args[0],
            f'--task={self.cli_task_name}',
            '--headless',
            '--seed=-1',
            f'--num_envs={self.task_num_envs}',
        ]

        if self.camera_enabled:
            command.append('--enable_cameras')

        command += modified_args[1:]
        
        if self.distributed:
            command += ['--distributed']

        print('Running command:', command, flush=True)
        print('sys.executable = ', sys.executable)
        print(f'Policy {self.policy_idx}: Restarting self with args {modified_args}', flush=True)
        
        
        with open('command.txt', 'a') as f:
            f.write(str(cli_args))
            f.write('\n')
            f.write(str(command))
            f.write('\n')
        
        if self.global_rank == 0:
            dump_env_sizes()
            
            # after any sourcing (or before exec’ing python.sh) prevent kept increasing arg_length:
            for var in ("PATH", "PYTHONPATH", "LD_LIBRARY_PATH",
                        "OMNI_USD_RESOLVER_MDL_BUILTIN_PATHS"):
                val = os.environ.get(var)
                if not val or os.pathsep not in val:
                    continue
                parts = val.split(os.pathsep)
                seen = set()
                new_parts = []
                for p in parts:
                    if p and p not in seen:
                        seen.add(p)
                        new_parts.append(p)
                os.environ[var] = os.pathsep.join(new_parts)

            os.execv(f'{isaac_sim_path}/python.sh', command)

def dump_env_sizes():
    # number of env vars
    n = len(os.environ)
    # total bytes in "KEY=VAL\0" for all envp entries
    total = sum(len(k) + 1 + len(v) + 1 for k, v in os.environ.items())
    # find the 5 largest values
    biggest = sorted(os.environ.items(), key=lambda kv: len(kv[1]), reverse=True)[:5]

    print(f"[ENV MONITOR] vars={n}, total_bytes={total}")
    for k, v in biggest:
        print(f"    {k!r} length={len(v)} → {v[:60]}{'…' if len(v) > 60 else ''}")

    try:
        argmax = os.sysconf('SC_ARG_MAX')
        print(f"[ENV MONITOR] SC_ARG_MAX = {argmax}")
    except (ValueError, AttributeError):
        pass