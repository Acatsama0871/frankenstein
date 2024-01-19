#!/bin/bash

dvc pull
zstd -d data/01_raw/outfits.tar.zst -o data/01_raw/outfits.tar
tar -xf data/01_raw/outfits.tar
