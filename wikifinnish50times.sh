#!/usr/bin/env bash
for((i=0; i<50; ++i));
do
    ./repfrench.py > wikifinnish_execution_$i &
done
