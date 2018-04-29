#!/usr/bin/env bash
for((i=0; i<50; ++i));
do
    ./repfrench.py > repfrench_execution_$i &
done
