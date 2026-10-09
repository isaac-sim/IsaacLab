* Fixed image transfer views drifting apart after one view reset mid-chunk, which filled its control queue: a
  resetting view now starts its episode at the next chunk with its newest control.
* Fixed image transfer views receiving extra frames when environments reset independently with a camera slower
  than the environment step: a capture now queues controls only for the views it covers.
