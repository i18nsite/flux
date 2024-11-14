#!/usr/bin/env bash

DIR=$(realpath $0) && DIR=${DIR%/*}
set -ex

cd $DIR/hit_sr
cp -f ../scale.py .
exec ./scale.py
