![Example Tasks created with ORBIT](docs/source/_static/tasks.jpg)

---

# Omniverse Isaac Orbit

[![IsaacSim](https://img.shields.io/badge/Isaac%20Sim-2022.2.0-orange.svg)](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://docs.python.org/3/whatsnew/3.7.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-lightgrey.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![Docs status](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://isaac-orbit.github.io/orbit)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)

<!-- TODO: Replace docs status with workflow badge? Link: https://github.com/isaac-orbit/orbit/actions/workflows/docs.yaml/badge.svg -->

Isaac Orbit (or *orbit* in short) is a unified and modular framework for robot learning powered by [NVIDIA Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html). It offers a modular design to easily and efficiently create robotic environments with photo-realistic scenes and fast and accurate simulation.

Please refer to our [documentation page](https://isaac-orbit.github.io/orbit) to learn more about the installation steps, features, and tutorials.

## ⚠️ Annoucement (22.09.2023)

We are currently in a phase of heavy development, and our team is actively working on various aspects of the framework to enhance its modularity and overall functionality. We understand the anticipation for a new release and assure you that we are working diligently towards it. While we have yet to set an exact release date to share, we are targeting a release in early October 2023. We believe that the improvements we are making will be well worth the wait.

For more details, please check the post here: https://github.com/NVIDIA-Omniverse/Orbit/discussions/106

---

## Contributing to Orbit

We wholeheartedly welcome contributions from the community to make this framework mature and useful for everyone. These may happen in
form of bug reports, feature requests, or code contributions. For details, please check our [contribution guidelines](https://isaac-orbit.github.io/orbit/source/refs/contributing.html).

## Troubleshooting

Please see the [troubleshooting](https://isaac-orbit.github.io/orbit/source/refs/troubleshooting.html) section for common fixes or [submit an issue](https://github.com/NVIDIA-Omniverse/orbit/issues).

For issues related to Isaac Sim, we recommend checking its [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html), or opening a question on its [forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/67).

## Support

* Please use GitHub [Discussions](https://github.com/NVIDIA-Omniverse/Orbit/discussions) for discussing ideas, asking questions, and requests for new features.
* Github [Issues](https://github.com/NVIDIA-Omniverse/orbit/issues) should only be used to track executable pieces of work with a definite scope and a clear deliverable. These can be fixing bugs, documentation issues, new features, or general updates.

## Acknowledgement

NVIDIA Isaac Sim is available freely under [individual license](https://www.nvidia.com/en-us/omniverse/download/). For more information about its license terms, please check [here](https://docs.omniverse.nvidia.com/app_isaacsim/common/NVIDIA_Omniverse_License_Agreement.html#software-support-supplement).

ORBIT framework is released under [BSD-3 License](LICENSE). The license files of its dependencies and assets are present in the [`docs/licenses`](docs/licenses) directory.

## Citation

Please cite [this paper](https://arxiv.org/abs/2301.04195) if you use this framework in your work:

```
@misc{mittal2023orbit,
	author = {Mayank Mittal and Calvin Yu and Qinxi Yu and Jingzhou Liu and Nikita Rudin and David 	Hoeller and Jia Lin Yuan and Pooria Poorsarvi Tehrani and Ritvik Singh and Yunrong Guo and Hammad Mazhar and Ajay Mandlekar and Buck Babich and Gavriel State and Marco Hutter and Animesh Garg},
	title = {ORBIT: A Unified Simulation Framework for Interactive Robot Learning Environments},
	year = {2023},
	eprint = {arXiv:2301.04195},
}
```
