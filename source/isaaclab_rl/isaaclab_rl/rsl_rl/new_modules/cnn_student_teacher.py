# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation


class CNNStudentTeacher(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
        num_actions,
        student_hidden_dims=[256, 256, 256],
        teacher_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=0.1,
        student_cnn_kernel_size=3,
        student_cnn_stride=3,
        student_cnn_filters=[32, 16, 8],
        student_paddings=[0, 0, 0],
        teacher_cnn_kernel_size=3,
        teacher_cnn_stride=3,
        teacher_cnn_filters=[32, 16, 8],
        teacher_paddings=[0, 0, 0],
        **kwargs,
    ):
        if kwargs:
            print(
                "StudentTeacher.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        activation = resolve_nn_activation(activation)
        self.loaded_teacher = False  # indicates if teacher has been loaded

        mlp_input_dim_s = num_student_obs
        mlp_input_dim_t = num_teacher_obs

        # student
        s_out_channels = student_cnn_filters
        s_in_channels = [1] + student_cnn_filters[:-1]
        student_layers = []
        mlp_input_dim_s = num_student_obs
        for in_ch, out_ch, pad in zip(s_in_channels, s_out_channels, student_paddings):
            student_layers.append(nn.Conv1d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=student_cnn_kernel_size,
                stride=student_cnn_stride,
                padding=pad
            ))
            student_layers.append(activation)
            mlp_input_dim_s = (mlp_input_dim_s + 2 * pad - student_cnn_kernel_size) // student_cnn_stride + 1

        student_layers.append(nn.Flatten())
        mlp_input_dim_s = mlp_input_dim_s * s_out_channels[-1]

        student_layers.append(nn.Linear(mlp_input_dim_s, student_hidden_dims[0]))
        student_layers.append(activation)
        for layer_index in range(len(student_hidden_dims)):
            if layer_index == len(student_hidden_dims) - 1:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], num_actions))
            else:
                student_layers.append(nn.Linear(student_hidden_dims[layer_index], student_hidden_dims[layer_index + 1]))
                student_layers.append(activation)
        self.student = nn.Sequential(*student_layers)

        # teacher
        t_out_channels = teacher_cnn_filters
        t_in_channels = [1] + teacher_cnn_filters[:-1]
        teacher_layers = []
        mlp_input_dim_t = num_teacher_obs
        for in_ch, out_ch, pad in zip(t_in_channels, t_out_channels, teacher_paddings):
            teacher_layers.append(nn.Conv1d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=teacher_cnn_kernel_size,
                stride=teacher_cnn_stride,
                padding=pad
            ))
            teacher_layers.append(activation)
            mlp_input_dim_t = (mlp_input_dim_t + 2 * pad - teacher_cnn_kernel_size) // teacher_cnn_stride + 1

        teacher_layers.append(nn.Flatten())
        mlp_input_dim_t = mlp_input_dim_t * s_out_channels[-1]

        teacher_layers.append(nn.Linear(mlp_input_dim_t, teacher_hidden_dims[0]))
        teacher_layers.append(activation)
        for layer_index in range(len(teacher_hidden_dims)):
            if layer_index == len(teacher_hidden_dims) - 1:
                teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], num_actions))
            else:
                teacher_layers.append(nn.Linear(teacher_hidden_dims[layer_index], teacher_hidden_dims[layer_index + 1]))
                teacher_layers.append(activation)
        self.teacher = nn.Sequential(*teacher_layers)
        self.teacher.eval()

        print(f"Student MLP: {self.student}")
        print(f"Teacher MLP: {self.teacher}")

        # action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    def reset(self, dones=None, hidden_states=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.student(observations.unsqueeze(1))
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations):
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        actions_mean = self.student(observations.unsqueeze(1))
        return actions_mean

    def evaluate(self, teacher_observations):
        with torch.no_grad():
            actions = self.teacher(teacher_observations.unsqueeze(1))
        return actions

    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the student and teacher networks.

        Args:
            state_dict (dict): State dictionary of the model.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the keys returned by this
                           module's state_dict() function.

        Returns:
            bool: Whether this training resumes a previous training. This flag is used by the `load()` function of
                  `OnPolicyRunner` to determine how to load further parameters.
        """

        # check if state_dict contains teacher and student or just teacher parameters
        if any("actor" in key for key in state_dict.keys()):  # loading parameters from rl training
            # rename keys to match teacher and remove critic parameters
            teacher_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value

            self.teacher.load_state_dict(teacher_state_dict)
                
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            return False
        elif any("student" in key for key in state_dict.keys()):  # loading parameters from distillation training
            super().load_state_dict(state_dict, strict=strict)
            # set flag for successfully loading the parameters
            self.loaded_teacher = True
            self.teacher.eval()
            return True
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass
