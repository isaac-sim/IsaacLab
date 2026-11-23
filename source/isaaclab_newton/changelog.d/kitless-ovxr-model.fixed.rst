* Fixed Newton CUDA graph capture conflicting with concurrent legacy-stream uploads, such as renderer
  or input uploads, by capturing on a nonblocking stream.
