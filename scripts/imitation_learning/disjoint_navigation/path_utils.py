# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch


def nearest_point_on_segment(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    a2b = b - a
    a2c = c - a
    a2b_mag = torch.sqrt(torch.sum(a2b**2))
    a2b_norm = a2b / (a2b_mag + 1e-6)
    dist = torch.dot(a2c, a2b_norm)
    if dist < 0:
        return a, dist
    elif dist > a2b_mag:
        return b, dist
    else:
        return a + a2b_norm * dist, dist


class PathHelper:
    def __init__(self, points: torch.Tensor):
        self.points = points
        self._init_point_distances()

    def _init_point_distances(self):
        self._point_distances = torch.zeros(len(self.points))
        length = 0.0
        for i in range(0, len(self.points) - 1):
            self._point_distances[i] = length
            a = self.points[i]
            b = self.points[i + 1]
            dist = torch.sqrt(torch.sum((a - b) ** 2))
            length += dist
        self._point_distances[-1] = length

    def point_distances(self):
        return self._point_distances

    def get_path_length(self):
        length = 0.0
        for i in range(1, len(self.points)):
            a = self.points[i - 1]
            b = self.points[i]
            dist = torch.sqrt(torch.sum((a - b) ** 2))
            length += dist
        return length

    def points_x(self):
        return self.points[:, 0]

    def points_y(self):
        return self.points[:, 1]

    def get_segment_by_distance(self, distance):

        for i in range(0, len(self.points) - 1):
            d_a = self._point_distances[i]
            d_b = self._point_distances[i + 1]

            if distance < d_b:
                return (i, i + 1)

        i = len(self.points) - 2

        return (i, i + 1)

    def get_point_by_distance(self, distance):
        a_idx, b_idx = self.get_segment_by_distance(distance)
        a, b = self.points[a_idx], self.points[b_idx]
        a_dist, b_dist = self._point_distances[a_idx], self._point_distances[b_idx]
        u = (distance - a_dist) / ((b_dist - a_dist) + 1e-6)
        u = torch.clip(u, 0.0, 1.0)
        return a + u * (b - a)

    def find_nearest(self, point):
        min_pt_dist_to_seg = 1e9
        min_pt_seg = None
        min_pt = None
        min_pt_dist_along_path = None
        
        for a_idx in range(0, len(self.points) - 1):
            b_idx = a_idx + 1
            a = self.points[a_idx]
            b = self.points[b_idx]
            nearest_pt, dist_along_seg = nearest_point_on_segment(a, b, point)
            dist_to_seg = torch.sqrt(torch.sum((point - nearest_pt) ** 2))

            if dist_to_seg < min_pt_dist_to_seg:
                min_pt_seg = (a_idx, b_idx)
                min_pt_dist_to_seg = dist_to_seg
                min_pt = nearest_pt
                min_pt_dist_along_path = self._point_distances[a_idx] + dist_along_seg

        return min_pt, min_pt_dist_along_path, min_pt_seg, min_pt_dist_to_seg
