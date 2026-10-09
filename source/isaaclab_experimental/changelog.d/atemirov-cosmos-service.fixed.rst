* Fixed image transfer views drifting apart after one view reset mid-chunk, which filled its control queue: a
  resetting view now starts its episode at the next chunk with its newest control.
