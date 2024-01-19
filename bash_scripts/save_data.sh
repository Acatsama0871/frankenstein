#!/bin/bash

tar --zstd -cf data/01_raw/outfits.tar.zst data/01_raw/outfits/
dvc add data
dvc push
