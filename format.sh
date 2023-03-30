#!/bin/bash
shopt -s globstar

for f in $(/bin/ls -1 **/*.ipynb); do /bin/cat <<< $(jq --indent 1 --monochrome-output '(.cells[] | select(has("outputs")) | .outputs) = []  | (.cells[] | select(has("execution_count")) | .execution_count) = null  | .metadata = {"language_info": {"name": "python", "pygments_lexer": "ipython3"}} | .cells[].metadata = {}' $f) > $f; done

nbqa black **/*.ipynb --nbqa-mutate
