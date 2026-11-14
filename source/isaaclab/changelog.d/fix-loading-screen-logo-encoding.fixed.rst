* Fixed the startup loading screen crashing with ``UnicodeEncodeError`` on consoles that cannot encode
  block or box-drawing characters, such as Windows output redirected to a file. The run summary drew a
  greeting logo picked at random without checking the console encoding; logos the console cannot encode
  are now left out.
