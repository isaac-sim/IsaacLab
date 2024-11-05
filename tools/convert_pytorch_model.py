import torch
import io
import rl_games.algos_torch.flatten as flatten

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        torch.nn.Module.__init__(self)
        self._model = model
    def forward(self, inputs):
        actions, log_prob, outputs = self._model.act({"states": inputs}, 'policy')
        return outputs["mean_actions"]

model = torch.load("/home/nexus/Documents/repos/IsaacLab/logs/skrl/unitree_go1_flat/2024-10-18_10-54-45_ppo_torch/checkpoints/agent_355000.pt")

example = torch.ones(1, 46)
adapter = flatten.TracingAdapter(ModelWrapper(model["policy"]), example, allow_non_tensor=True)
traced = torch.jit.trace(adapter, adapter.flattened_inputs, check_trace=False)
traced.eval()
output = traced.forward(example)

torch.jit.save(m, 'policy_jit.pt')
