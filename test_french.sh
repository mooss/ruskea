#!/usr/bin/env bash
./repextract.py
pattern='.french_results'
./repfrench.py > $pattern
<<extractresults>>
