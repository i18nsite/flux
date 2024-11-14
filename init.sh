#!/usr/bin/env bash

DIR=$(realpath $0) && DIR=${DIR%/*}
cd $DIR
set -ex

if [ ! -d hit-sr ]; then
  git clone --depth=1 git@hf.co:XiangZ/hit-sr hit_sr
  cd hit_sr
  touch __init__.py
  pip install -r requirements.txt
fi

mise exec -- python -m venv .venv
mise exec -- pip install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu

# modelscope
mise exec -- pip install diffusers accelerate torch protobuf sentencepiece peft transformers[sentencepiece] pyyaml Pillow python-baseconv tqdm
