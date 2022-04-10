# Read Serial Data for Ergo

This simple rust script reads the serial data coming in from one of the `/dev/`
serial ports and sends it to `stdout`. Eventually it will do some light
pre-processing and write it to file, or alternatively provide visual feedback
on the stream of numbers coming through.
