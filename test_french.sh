#!/usr/bin/env bash
echo "testing on est républicain corpus"
./repextract.py
pattern='.french_results'
./repfrench.py > $pattern
best=$(grep -nH score $pattern* | sort -k 2 -t ' ' -gr | head -n1)
file=$(echo $best | cut -f 1 -d ':')
line=$(echo $best | cut -f 2 -d ':')
tail --lines=+$((line + 1)) $file
