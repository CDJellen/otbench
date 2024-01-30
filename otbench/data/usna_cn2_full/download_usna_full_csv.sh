#!/bin/bash

URL="drive.google.com/u/3/uc?id=18O7s6Fz287Yo0iUyV_HASShoKUPFKsAB&export=download&confirm=yes"
FILENAME="usna_full.csv"

# Download the file
wget "$URL" -O "$FILENAME"
