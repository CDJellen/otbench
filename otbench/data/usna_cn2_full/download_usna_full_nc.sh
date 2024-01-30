#!/bin/bash

URL="drive.google.com/u/3/uc?id=1sio9lqMteIx67OVcUmRDH1Ry7pe2i48F&export=download&confirm=yes"
FILENAME="usna_full.nc"

# Download the file
wget "$URL" -O "$FILENAME"
