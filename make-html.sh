#!/bin/bash -eu


# For newer directories
for NBPATH in $(shopt -s globstar && /bin/ls -1 notebooks/**/examples/*.ipynb notebooks/**/tutorials/*.ipynb); do
    DIR=html/$(echo ${NBPATH} | cut -d / -f 2-3)/
    OUTPUT=$(jupyter nbconvert --embed-images --output-dir=$DIR --to html "$NBPATH")
    if [ -n "$OUTPUT" ]; then
        echo "[NbConvertApp] Converting notebook $NBPATH to html" 1>&2
        echo "$OUTPUT" 1>&2
    fi
done
