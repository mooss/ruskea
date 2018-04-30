#!/usr/bin/env bash
echo "testing on brown corpus"
./brownextract.py > /dev/null
echo "predefined matrices groups"
./brownmarvin.py | tail -n 1
echo "procedurally generated groups"
./brownrandom.py | tail -n 1
