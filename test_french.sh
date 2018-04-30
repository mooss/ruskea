#!/usr/bin/env bash
echo "testing on est républicain corpus"
./repextract.py
pattern='.french_results'
./repfrench.py > $pattern
<<extractresults>>
