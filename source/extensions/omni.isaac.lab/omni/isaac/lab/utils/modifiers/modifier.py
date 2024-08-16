# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from collections.abc import Sequence

from .modifier_base import ModifierBase
from .modifier_cfg import ModifierCfg

##
# Modifiers as functions
##


def scale(data: torch.Tensor, multiplier: float) -> torch.Tensor:
    """Scales input data by a multiplier.

    Args:
        data: The data to apply the scale to.
        multiplier: Value to scale input by.

    Returns:
        Scaled data. Shape is the same as data.
    """
    return data * multiplier


def clip(data: torch.Tensor, bounds: tuple[float | None, float | None]) -> torch.Tensor:
    """Clips the data to a minimum and maximum value.

    Args:
        data: The data to apply the clip to.
        bounds: A tuple containing the minimum and maximum values to clip data to.
            If the value is None, that bound is not applied.

    Returns:
        Clipped data. Shape is the same as data.
    """
    return data.clip(min=bounds[0], max=bounds[1])


def bias(data: torch.Tensor, value: float) -> torch.Tensor:
    """Adds a uniform bias to the data.

    Args:
        data: The data to add bias to.
        value: Value of bias to add to data.

    Returns:
        Biased data. Shape is the same as data.
    """
    return data + value


##
# Sample of class based modifiers
##


class DigitalFilter(ModifierBase):
    r"""Modifier used to apply digital filtering using the linear difference form of a discrete z-transform filter definition.

    **Z-transform:**

    .. math::
        \begin{equation}
            H(z)
            = \frac{Y(z)}{X(z)}
            = \frac{b_{0} + b_{1}z^{-1} + b_{2}z{^-2} ... b_{N}z^{-N}}{1 + a_{1}z^{-1} + a_{2}z^{-2} ... a_{M}z^{-M}}
        \end{equation}

    **Linear difference form:**

    .. math::
         y_{i} &= XB - YA \\
               &= \begin{bmatrix} x_{i}&x_{i-1}&...&x_{i-N} \end{bmatrix} \begin{bmatrix} b_{0} \\ b_{1} \\ ... \\ b_{N}\end{bmatrix}
               - \begin{bmatrix} y_{i-1}&y_{i-2}&...&y_{i-M} \end{bmatrix} \begin{bmatrix} a_{1} \\ a_{2} \\ ... \\ a_{M}\end{bmatrix}

    where :math:`y_{i}` is the output of the filter. :math:`Y` is a tensor containing previous outputs of the filter :math:`y_{i-M}` for :math:`M` previous timesteps
    and :math:`X` is a tensor containing current, :math:`x_{i}`, and previous inputs to the filter :math:`x_{i-N}` for :math:`N` previous timesteps.
    :math:`A` and :math:`B`, with length :math:`M` and :math:`N` are vectors of denominator and numerator coefficients respectively. Choosing :math:`A` and :math:`B`
    is up to the user. Examples below will show how to setup filter coefficients for common filters like a first order IIR low-pass filter
    and a unit delay function.

    **Example: First order IIR low-pass filter**

    Because digital filters act on discrete signals, the timestep, :math:`\Delta t` is used to calculate desired cut-off frequencies
    of filters. For a first order IIR lowpass filter the transfer function equation simplifies to:

    .. math::   \begin{equation}H(z)=\frac{\alpha}{1-\alpha}\end{equation}

    with accompanying linear difference formulation:

    .. math:: y_{i} = \alpha x_{i} + (1-\alpha)y_{i-1}

    where :math:`\alpha` is a smoothing parameter calculated from desired cut-off frequency, :math:`f_{c}` in Hz:

    .. math::   \alpha = \frac{f_{c}}{f_{c} + \frac{1}{2\pi\Delta t}}

    In order to implement this filter with the :class:`DigitalFilter` follow these steps in your :class:`ModifierCfg`:

    .. code-block:: python

        import math
        import torch
        from omni.isaac.lab.utils import modifiers

        # with sim.physics_step = 0.002
        fc_hz = 20 # desired cut-off frequency
        alpha = fc_hz/(fc_hz + 1/(2.0*math.pi*0.002) # desired smoothing parameter
        # create filter and filter coefficients for first order IIR low-pass filter
        my_modifier_cfg = modifiers.ModifierCfg(func=modifiers.DigitalFilter, params={"A": torch.tensor([1.0-alpha]), "B": torch.tensor([alpha])})
        # create class instance
        my_filter = modifiers.Modifier(cfg=my_modifier_cfg)

    **Example: Unit delay**

    The same digital filter implementation can be used to do a unit delay. In this case there is not filtering history that needs to be kept.
    This will only utilize the previous measurements :math:`x_{i-n}`.

    In the case of a single timestep unit delay the linear difference equation simplifies to:

    .. math:: y_{i} = x_{i-1}

    Implementation looks like:

    .. code-block:: python

        import torch
        from omni.isaac.lab.utils import modifiers

        # create single timestep unit delay config
        my_modifier_cfg = modifiers.ModifierCfg(func=modifiers.DigitalFilter, params={"A": torch.tensor([0.0]), "B": torch.tensor([0.0, 1.0])})

        # create class instance
        my_delay = modifiers.Modifier(cfg=my_modifier_cfg)

    .. warning:: The filter coefficients :math:`A` and :math:`B` must not be None.


    Extra explanation on digital filters and other filter types can be found at: https://en.wikipedia.org/wiki/Digital_filter
    """

    def __init__(self, cfg: ModifierCfg, data_dim: tuple[int, ...], device: str) -> None:
        """Initializes digital filter.

        Args:
            cfg: Configuration parameters.
            data_dim: The dimensions of the data to be modified. First element is the batch size
                which usually corresponds to number of environments in the simulation.
            device: The device to run the modifier on.

        Raises:
            ValueError: If filter coefficients are None.
        """
        # initialize parent class
        super().__init__(cfg, data_dim, device)

        # check if params has the required coefficients
        if "A" not in self._cfg.params or "B" not in self._cfg.params:
            raise ValueError("Digital filter configuration must have 'A' and 'B' keys in the params dictionary.")

        # assign filter coefficients
        self.A = self._cfg.params["A"]
        self.B = self._cfg.params["B"]

        # check that filter coefficients are not None
        if self.A is None or self.B is None:
            raise ValueError("Digital filter coefficients A and B must not be None. Please provide valid coefficients.")

        # create buffer for input and output history
        self.x_n = torch.zeros(self._data_dim + (self.B.shape[0],), device=self._device)
        self.y_n = torch.zeros(self._data_dim + (self.A.shape[0],), device=self._device)

    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets digital filter history.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        if env_ids is None:
            self.x_n.zero_()
            self.y_n.zero_()
        else:
            self.x_n[env_ids].zero_()
            self.y_n[env_ids].zero_()

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Applies digital filter modification with a rolling history window inputs and outputs.

        Args:
            data: The data to apply filter to.

        Returns:
            Filtered data. Shape is the same as data.
        """
        # move history window for input
        self.x_n = torch.roll(self.x_n, shifts=1, dims=-1)
        self.x_n[..., 0] = data

        # make coefficients column vectors
        A = self.A.unsqueeze(1)
        B = self.B.unsqueeze(1)

        # calculate current filter value: y[i] = Y*A - X*B
        y_i = torch.matmul(self.x_n, B) - torch.matmul(self.y_n, A)
        y_i.squeeze_(-1)

        # move history window for output and add current filter value to history
        self.y_n = torch.roll(self.y_n, shifts=1, dims=-1)
        self.y_n[..., 0] = y_i

        return y_i


class Integrator(ModifierBase):
    """Modifier that applies a numerical forward integration based on a middle Reimann sum.

    **Usage:**

    .. code-block:: python

        from omni.isaac.lab.utils import modifiers

        dt = 0.001 # time step in seconds between data measurements
        # create Integrator config
        my_modifier_cfg = modifiers.ModifierCfg(func=modifiers.Integrator, params={"dt":dt})

        # create class instance
        my_integrator = modifiers.Modifier(cfg=my_modifier_cfg)

    Reference: https://en.wikipedia.org/wiki/Riemann_sum
    """

    def __init__(self, cfg: ModifierCfg, data_dim: tuple[int, ...], device: str):
        """Initializes the integrator configuration and state.

        Args:
            cfg: Integral parameters.
            data_dim: The dimensions of the data to be modified. First element is the batch size
                which usually corresponds to number of environments in the simulation.
            device: The device to run the modifier on.
        """
        # initialize parent class
        super().__init__(cfg, data_dim, device)

        # check if params has the required dt key
        if "dt" not in self._cfg.params:
            raise ValueError("Integrator configuration must have a 'dt' key in the params dictionary.")

        # assign buffer for integral and previous value
        self.integral = torch.zeros(self._data_dim, device=self._device)
        self.y_prev = torch.zeros(self._data_dim, device=self._device)
        # store time step
        self.dt = self._cfg.params["dt"]

    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets integrator state to zero.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.
        """
        if env_ids is None:
            self.integral.zero_()
            self.y_prev.zero_()
        else:
            self.integral[env_ids].zero_()
            self.y_prev[env_ids].zero_()

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Applies integral modification to input data.

        Args:
            data: The data to integrate.

        Returns:
            Integral of input signal. Shape is the same as data.
        """
        # integrate using middle Riemann sum
        self.integral += (data + self.y_prev) / 2 * self.dt
        # update previous value
        self.y_prev[:] = data

        return self.integral
