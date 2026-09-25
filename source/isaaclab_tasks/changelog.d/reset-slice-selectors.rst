Fixed
^^^^^

* Handled full and strided reset slices in sampled deformable observations and deployment noise models.
* Used existing device indices for multitask reset slices without implicitly converting caller-provided indices.
* Preserved full and partial slices in task reset events, curricula, and commands, using selected data
  shapes or slice bounds when only a batch size was required.
