# Allegro Cache Score Guide

Use cache diagnostics to decide whether a generated cache is worth training from.

Good signs:

```text
stable is high,
contact_fingers is about 3 or higher,
thumb_contact is high,
near is close to 4,
object height remains in the reset window.
```

Bad signs:

```text
cache size stays at zero,
thumb_contact is low,
contact_fingers is below 3,
object falls below the reset window,
object is held by only one or two fingers.
```

If the cache score is poor, adjust the ready pose or reset offsets before training.
