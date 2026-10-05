* Fixed the feet-wrench observation of ``Isaac-Humanoid`` and ``Isaac-Humanoid-Direct`` listing the feet in the
  joint-wrench sensor's own body order, which is ``[right_foot, left_foot]`` on PhysX and ``[left_foot, right_foot]``
  on Newton. The observation now follows ``feet_body_names`` on every backend. A policy trained on PhysX with the old
  layout sees the feet swapped, so retrain it or swap the two six-value feet blocks of its observation.
