#!/bin/bash

set -e

DIR="./logs"

if [ -d "$DIR" ]; then
    rm -rf "$DIR"/* "$DIR"/.[!.]* "$DIR"/..?*
else
    echo "logs/ does not exist"
fi
