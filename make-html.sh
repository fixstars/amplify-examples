#!/bin/bash -eu

for NBPATH in $(shopt -s globstar && /bin/ls -1 notebooks/**/samples/*.ipynb); do
    JP_OR_EN=$(echo ${NBPATH} | cut -c 11-12)
    DIR=./html/${JP_OR_EN}/
    OUTPUT=$(jupyter nbconvert --embed-images --output-dir=$DIR --to html "$NBPATH")
    if [ -n "$OUTPUT" ]; then
        echo "[NbConvertApp] Converting notebook $NBPATH to html" 1>&2
        echo "$OUTPUT" 1>&2
    fi
done

