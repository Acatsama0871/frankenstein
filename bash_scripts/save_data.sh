#!/bin/bash

tar -cf - data/01_raw/outfits | zstd -o data/01_raw/outfits.tar.zst
dvc add data
dvc push
