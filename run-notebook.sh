#!/bin/bash -eu

: "${AMPLIFYAECLIENT_URL:?undefined}"
: "${AMPLIFYAECLIENT_TOKEN:?undefined}"

echo "[run-notebook] Executing notebook $1"

NBTMP=$(mktemp -p "$(dirname "$1")" .ci-exec-XXXXXXXX.ipynb)
trap 'rm -f "$NBTMP"' EXIT

jq --monochrome-output '
    (.cells[] | select(.cell_type == "code") | .source) |=
    ((if type == "string" then [splits("(?<=\n)")] else . end) | map(
        if test("^[ \t]*#[ \t]*client\\.token[ \t]*=")
        then sub("#.*"; "import os; client.url = os.getenv(\"AMPLIFYAECLIENT_URL\", client.url); client.token = os.getenv(\"AMPLIFYAECLIENT_TOKEN\", client.token)")
        else . end))
' "$1" > "$NBTMP"

if grep -qE '(AmplifyAEClient|FixstarsClient) *\(' "$1" && ! grep -q AMPLIFYAECLIENT_TOKEN "$NBTMP"; then
    echo "[run-notebook] failed to inject credentials into $1 (no commented-out client.token line?)" 1>&2
    exit 1
fi

uv run jupyter nbconvert --to html --stdout --execute "$NBTMP" --log-level WARN > /dev/null \
    || { echo "[run-notebook] failed to run: $1" 1>&2; exit 1; }
