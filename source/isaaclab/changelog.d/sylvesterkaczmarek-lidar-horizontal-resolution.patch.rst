Bug Fixes
---------
* Preserved the configured LiDAR horizontal angular resolution when the field-of-view span is not evenly divisible by it.
* For non-divisible spans, the generated horizontal ray count may decrease because an off-grid terminal endpoint is no longer forced into the pattern. This can change ``RayCaster.num_rays`` and downstream observation dimensions. To retain endpoint-inclusive sampling, choose a horizontal resolution that evenly divides the field-of-view span.
