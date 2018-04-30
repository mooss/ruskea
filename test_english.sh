#!/usr/bin/env bash
echo "testing on brown corpus"
./brownextract.py > /dev/null
echo "predefined matrices groups "
./brownmarvin.py | tail -n 1
./brownrandom.py | tail -n 1
