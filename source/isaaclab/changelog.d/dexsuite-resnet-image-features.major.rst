Changed
^^^^^^^

* Improved :class:`~isaaclab.envs.mdp.observations.image_features` ResNet model
  preparation to extract feature vectors instead of classification logits,
  resize inputs to the ImageNet pretraining resolution by default, use current
  torchvision weights enums, and support optional unfrozen gradient flow.
  **Breaking:** Policies trained on the previous 1000-dimensional ResNet logits
  output are incompatible and must be retrained.
