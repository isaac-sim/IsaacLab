from isaaclab_rl.rsl_rl.exporter import _TorchPolicyExporter, _OnnxPolicyExporter
import os, torch, copy
from typing import Any

def export_teacher_policy_as_jit(policy: object, normalizer: object | None, path: str, filename="policy.pt"):
    policy_exporter = _ParkourTeacherTorchPolicyExporter(policy, normalizer)
    policy_exporter.export(path, filename)

def export_teacher_policy_as_onnx(
    policy: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _ParkourTeacherOnnxPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)


def export_deploy_policy_as_jit(
    policy: object, 
    estimator: object, 
    depth_encoder: object, 
    normalizer: object | None, 
    path: str, 
    filename="policy.pt"
    ):
    policy_exporter = _ParkourDeployTorchPolicyExporter(policy, estimator, normalizer)
    policy_exporter.export(path, filename)

    depth_exporter = _ParkourDeployTorchDepthEncoderExporter(depth_encoder)
    depth_exporter.export(path, 'depth_latest.pt')

def export_deploy_policy_as_onnx(
    policy: object, 
    estimator: object, 
    depth_encoder: object, 
    agent_cfg: Any,
    path: str, 
    normalizer: object | None = None, 
    filename="policy.onnx", 
    verbose=False
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _ParkourDeployOnnxPolicyExporter(policy, estimator, normalizer, verbose)
    policy_exporter.export(path, filename)

    depth_exporter = _ParkourDeployOnnxDepthEncoderExporter(depth_encoder, agent_cfg, verbose)
    depth_exporter.export(path, 'depth_latest.onnx')

class _ParkourTeacherTorchPolicyExporter(_TorchPolicyExporter):
    def __init__(self, policy, normalizer=None):
        super().__init__(policy, normalizer)

    def forward_lstm(self, x):
        x = self.normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        x = x.squeeze(0)
        return self.actor(x, hist_encoding=True)
    
    def forward(self, x):
        return self.actor(self.normalizer(x), hist_encoding=True)
    
class _ParkourTeacherOnnxPolicyExporter(_OnnxPolicyExporter):
    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__(policy, normalizer, verbose)

    def forward_lstm(self, x_in, h_in, c_in):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        return self.actor(x, hist_encoding=True ), h, c

    def forward(self, x):
        return self.actor(self.normalizer(x), hist_encoding=True )

    def export(self, path, filename):
        self.to("cpu")
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            actions, h_out, c_out = self(obs, h_in, c_in)
            torch.onnx.export(
                self,
                (obs, h_in, c_in),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={},
            )
        else:
            obs = torch.zeros(1, self.actor.in_features)
            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )


class _ParkourDeployTorchPolicyExporter(torch.nn.Module):
    def __init__(self, policy, estimator ,normalizer=None):
        super().__init__()
        # copy policy parameters
        self.actor = copy.deepcopy(policy)
        self.estimator = copy.deepcopy(estimator)
        self._num_prop = policy.num_prop
        self._num_scan = policy.num_scan
        self._num_priv_explicit = policy.num_priv_explicit
        self._start = self._num_prop + self._num_scan 
        self._end = self._start + self._num_priv_explicit
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, x, scandots_latent):
        x[:,self._start : self._end] \
            = self.estimator(x[:,:self._num_prop])
        return self.actor(self.normalizer(x), hist_encoding=True , scandots_latent = scandots_latent)
    
    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

class _ParkourDeployOnnxPolicyExporter(torch.nn.Module):
    def __init__(self, policy, estimator ,normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        # copy policy parameters
        self.actor = copy.deepcopy(policy)
        self.estimator = copy.deepcopy(estimator)
        self._num_prop = policy.num_prop
        self._num_priv_explicit = policy.num_priv_explicit
        self._num_scan = policy.num_scan
        self._start = policy.num_prop + self._num_scan
        self._end = self._start + self._num_priv_explicit
        # set up recurrent network
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()

    def forward(self, x, scandots_latent):
        x[:,self._start : self._end] \
            = self.estimator(x[:,:self._num_prop])
        return self.actor(self.normalizer(x), hist_encoding=True, scandots_latent = scandots_latent )

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.actor.in_features)
        scandots_latent = torch.zeros(1, 32)
        torch.onnx.export(
            self,
            (obs, scandots_latent),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs","scandots_latent"],
            output_names=["actions"],
            dynamic_axes={},
        )

class _ParkourDeployTorchDepthEncoderExporter(torch.nn.Module):
    def __init__(self, depth_encoder):
        super().__init__()
        # copy policy parameters
        self.depth_encoder = copy.deepcopy(depth_encoder)

    def forward(self, depth_image, proprioception):
        return self.depth_encoder(depth_image, proprioception)
    
    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

class _ParkourDeployOnnxDepthEncoderExporter(torch.nn.Module):
    def __init__(self, depth_encoder, agent_cfg, verbose=False):
        super().__init__()
        self.verbose = verbose
        # copy policy parameters
        self.depth_encoder = copy.deepcopy(depth_encoder)
        self._image_size = agent_cfg.depth_encoder.depth_shape
        self._num_prop = agent_cfg.estimator.num_prop
        
    def forward(self, depth_image, proprioception):
        return self.depth_encoder(depth_image, proprioception)

    def export(self, path, filename):
        self.to("cpu")
        depth_image = torch.zeros(1, *self._image_size)
        proprioception = torch.zeros(1, self._num_prop)
        torch.onnx.export(
            self,
            (depth_image, proprioception),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["depth_image","proprioception"],
            output_names=["depth_latent_and_yaw"],
            dynamic_axes={},
        )

