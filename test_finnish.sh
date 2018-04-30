#!/usr/bin/env bash
echo "testing on finnish wikipedia corpus"

# The wikipedia dump is too heavy to put on github or to download separately
# ./wikifiextract.sh

pattern='.finnish_results'
./wikifinnish.py > $pattern
<<extractresults>>
