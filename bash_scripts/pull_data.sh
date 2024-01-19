#!/bin/bash

dvc pull
zstd -d data/01_raw/outfits.tar.zst -o - | tar -xf -
