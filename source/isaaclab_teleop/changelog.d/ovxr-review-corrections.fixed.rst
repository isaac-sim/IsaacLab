* Fixed OVXR reset publication, failed-generation retirement, Capture reset notifications and cleanup ownership.
* Preserved simulation clocks, optional teleop imports and positional configuration arguments.

Replay retains failed Capture input owners on Kit as well as OVXR, preserves the original failure, and services independent CloudXR cleanup after other cleanup failures.

Live OVXR honors shutdown signals during input startup and keeps repeated signals from interrupting cleanup.
