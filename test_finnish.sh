#!/usr/bin/env bash
echo "testing on finnish wikipedia corpus"

# The wikipedia dump is too heavy to put on github or to download separately
# ./wikifiextract.sh

pattern='.finnish_results'
./wikifinnish.py > $pattern
best=$(grep -n score $pattern* | sort -k 2 -t ' ' -gr | head -n1)
file=$(echo $best | cut -f 1 -d ':')
line=$(echo $best | cut -f 2 -d ':')
tail --lines=+$((line + 1)) $file
