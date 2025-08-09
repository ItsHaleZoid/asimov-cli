#!/bin/bash

# Start the SSH daemon in the background
/usr/sbin/sshd -D &

# Increase file descriptor limit to mitigate 'Too many open files' under heavy IO
ulimit -n 65535 || true

# Keep the container running. This is a simple way to keep the container
# alive indefinitely. You can replace this with your actual training script
# if you want it to run automatically upon container start.
echo "Container is running. SSH server is active."
echo "Connect via 'vastai ssh <instance_id>'"

# This will keep the container alive.
tail -f /dev/null 