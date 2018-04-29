#!/usr/bin/env bash
best=$(grep -n score repfrench_execution_* | sort -k 2 -t ' ' -gr | head -n1)
file=$(echo $best | cut -f 1 -d ':')
line=$(echo $best | cut -f 2 -d ':')
tail --lines=+$((line + 1)) $file
