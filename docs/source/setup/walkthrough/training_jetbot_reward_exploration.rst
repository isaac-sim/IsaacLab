.. _walkthrough_training_jetbot_reward_exploration:

Exploring the RL problem
=========================

The command to the Jetbot is a unit vector in specifying the desired drive direction and we must make the agent aware of this somehow
so it can adjust its actions accordingly.  There are many possible ways to do this, with the "zeroth order" approach to simply change the observation space to include
this command. To start, **edit the ``IsaacLabTutorialEnvCfg`` to set the observation space to 9**: the world velocity vector contains the linear and angular velocities
of the robot, which is 6 dimensions and if we append the command to this vector, that's 9 dimensions for the observation space in total.

Next, we just need to do that appending when we get the observations.  We also need to calculate our forward vectors for later use. The forward vector for the Jetbot is
the x axis, so we apply the ``root_link_quat_w`` to ``[1,0,0]`` to get the forward vector in the world frame. Replace the ``_get_observations`` method with the following:

.. code-block:: python

    def _get_observations(self) -> dict:
        self.velocity = self.robot.data.root_com_vel_w
        self.forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)
        obs = torch.hstack((self.velocity, self.commands))
        observations = {"policy": obs}
        return observations

 So now what should the reward be?

When the robot is behaving as desired, it will be driving at full speed in the direction of the command. If we reward both
"driving forward" and "alignment to the command", then maximizing that combined signal should result in driving to the command... right?

Let's give it a try! Replace the ``_get_rewards`` method with the following:

.. code-block:: python

    def _get_rewards(self) -> torch.Tensor:
        forward_reward = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
        alignment_reward = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        total_reward = forward_reward + alignment_reward
        return total_reward

The ``forward_reward`` is the x component of the linear center of mass velocity of the robot in the body frame. We know that
the x direction is the forward direction for the asset, so this should be equivalent to inner product between the forward vector and
the linear velocity in the world frame.  The alignment term is the inner product between the forward vector and the command vector: when they are
pointing in the same direction this term will be 1, but in the opposite direction it will be -1.  We add them together to get the combined reward and
we can finally run training!  Let's see what happens!

.. code-block:: bash

    python scripts/skrl/train.py --task=Template-Isaac-Lab-Tutorial-Direct-v0


.. figure:: https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/walkthrough_naive_webp.webp
    :align: center
    :figwidth: 100%
    :alt: Naive results

Surely we can do better!

Reward and Observation Tuning
-------------------------------

When tuning an environment for training, as a rule of thumb, you want to keep the observation space as small as possible.  This is to
reduce the number parameters in the model (the literal interpretation of Occam's razor) and thus improve training time. In this case we
need to somehow encode our alignment to the command and our forward speed. One way to do this is to exploit the dot and cross products
from linear algebra! Replace the contents of ``_get_observations`` with the following:

.. code-block:: python

    def _get_observations(self) -> dict:
        self.velocity = self.robot.data.root_com_vel_w
        self.forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)

        dot = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        cross = torch.cross(self.forwards, self.commands, dim=-1)[:,-1].reshape(-1,1)
        forward_speed = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
        obs = torch.hstack((dot, cross, forward_speed))

        observations = {"policy": obs}
        return observations

We also need to **edit the ``IsaacLabTutorialEnvCfg`` to set the observation space back to 3** which includes the dot product, the z component of the cross product, and the forward speed.

The dot or inner product tells us how aligned two vectors are as a single scalar quantity.  If they are very aligned and pointed in the same direction, then the inner
product will be large and positive, but if they are aligned and in opposite directions, it will be large and negative.  If two vectors are
perpendicular, the inner product is zero. This means that the inner product between the forward vector and the command vector can tell us
how much we are facing towards or away from the command, but not which direction we need to turn to improve alignment.

The cross product also tells us how aligned two vectors are, but it expresses this relationship as a vector.  The cross product between any
two vectors defines an axis that is perpendicular to the plane containing the two argument vectors, where the direction of the result vector along this axis is
determined by the chirality (dimension ordering, or handedness) of the coordinate system. In our case, we can exploit the fact that we are operating in 2D to only
examine the z component of the result of :math:`\vec{forward} \times \vec{command}`. This component will be zero if the vectors are colinear, positive if the
command vector is to the left of forward, and negative if it's to the right.

Finally, the x component of the center of mass linear velocity tells us our forward speed, with positive being forward and negative being backwards. We stack these together
"horizontally" (along dim 1) to generate the observations for each Jetbot. This alone improves performance!


.. figure:: https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/walkthrough_improved_webp.webp
    :align: center
    :figwidth: 100%
    :alt: Improved results

It seems to qualitatively train better, and the Jetbots are somewhat inching forward... Surely we can do better still!

Another rule of thumb for training is to reduce and simplify the reward function as much as possible.  Terms in the reward behave similarly to
the logical "OR" operation.  In our case, we are rewarding driving forward and being aligned to the command by adding them together, so our agent
can be reward for driving forward OR being aligned to the command. To force the agent to learn to drive in the direction of the command, we should only
reward the agent driving forward AND being aligned. Logical AND suggests multiplication and therefore the following reward function:

.. code-block:: python

    def _get_rewards(self) -> torch.Tensor:
        forward_reward = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
        alignment_reward = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        total_reward = forward_reward*alignment_reward
        return total_reward

Now we will only get rewarded for driving forward if our alignment reward is non zero.  Let's see what kind of result this produces!

.. figure:: https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/walkthrough_tuned_webp.webp
    :align: center
    :figwidth: 100%
    :alt: Tuned results

It definitely trains faster, but the Jetbots have learned to drive in reverse if the command is pointed behind them. This may be desirable in our
case, but it shows just how dependent the policy behavior is on the reward function.  In this case, there are **degenerate solutions** to our
reward function: The reward is maximized for driving forward and aligned to the command, but if the Jetbot drives in reverse, then the forward
term is negative, and if its driving in reverse towards the command, then the alignment term is **also negative**, meaning hat the reward is positive!
When you design your own environments, you will run into degenerate solutions like this and a significant amount of reward engineering is devoted to
suppressing or supporting these behaviors by modifying the reward function.

Let's say, in our case, we don't want this behavior. In our case, the alignment term has a domain of ``[-1, 1]``, but we would much prefer it to be mapped
only to positive values. We don't want to *eliminate* the sign on the alignment term, rather, we would like large negative values to be near zero, so if we
are misaligned, we don't get rewarded. The exponential function accomplishes this!

.. code-block:: python

    def _get_rewards(self) -> torch.Tensor:
        forward_reward = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1)
        alignment_reward = torch.sum(self.forwards * self.commands, dim=-1, keepdim=True)
        total_reward = forward_reward*torch.exp(alignment_reward)
        return total_reward

Now when we train, the Jetbots will turn to always drive towards the command in the forward direction!

.. figure:: https://download.isaacsim.omniverse.nvidia.com/isaaclab/images/walkthrough_directed_webp.webp
    :align: center
    :figwidth: 100%
    :alt: Directed results
