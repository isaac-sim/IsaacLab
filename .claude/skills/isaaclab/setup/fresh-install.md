# Isaac Lab Installation

Refer to the official installation documentation — it covers all supported methods
and stays up to date with each release:

```
docs/source/setup/installation/index.rst
```

## Installation Methods (summary)

1. **Kit-less** (Newton only, no Isaac Sim): `./isaaclab.sh --install`
2. **Pip install** (recommended): Isaac Sim via pip + Isaac Lab from source
3. **Binary + source**: Isaac Sim binary + Isaac Lab source
4. **Full source build**: Both from source (for developers)
5. **Docker**: Containerized deployment

## Quick Start (Kit-less / Newton)

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
./isaaclab.sh -i all
```

## Quick Start (with Isaac Sim)

```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
pip install isaacsim
./isaaclab.sh -i all
```

## Install RL Frameworks

```bash
./isaaclab.sh -i rsl_rl
./isaaclab.sh -i sb3
./isaaclab.sh -i skrl
./isaaclab.sh -i rl_games
```

## Key Rule

Always use `./isaaclab.sh -p` to run Python scripts — never bare `python3`.
