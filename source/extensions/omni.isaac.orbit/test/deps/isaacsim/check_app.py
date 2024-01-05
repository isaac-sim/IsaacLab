# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script shows the issue with launching Isaac Sim application in headless mode.

On launching the application in headless mode, the application does not exit gracefully.
There are multiple warnings and errors that are printed on the console.

```
_isaac_sim/python.sh source/extensions/omni.isaac.orbit/test/deps/isaacsim/check_app.py
```

Output:

```
[10.948s] Simulation App Startup Complete
[11.471s] Simulation App Shutting Down
...... [Warning] [carb] [Plugin: omni.spectree.delegate.plugin] Module /media/vulcan/packman-repo/chk/kit-sdk/105.1+release.129498.98d86eae.tc.linux-x86_64.release/exts/omni.usd_resolver/bin/libomni.spectree.delegate.plugin.so remained loaded after unload request
...... [Warning] [omni.core.ITypeFactory] Module /media/vulcan/packman-repo/chk/kit-sdk/105.1+release.129498.98d86eae.tc.linux-x86_64.release/exts/omni.graph.action/bin/libomni.graph.action.plugin.so remained loaded after unload request.
...... [Warning] [omni.core.ITypeFactory] Module /media/vulcan/packman-repo/chk/kit-sdk/105.1+release.129498.98d86eae.tc.linux-x86_64.release/exts/omni.activity.core/bin/libomni.activity.core.plugin.so remained loaded after unload request.
```

"""

from __future__ import annotations

from omni.isaac.kit import SimulationApp

if __name__ == "__main__":
    app = SimulationApp({"headless": True})
    app.close()
