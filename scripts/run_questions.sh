#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 INDEX QUESTIONS OUTPUT" >&2
  exit 2
fi

quest run --index "$1" --questions "$2" --output "$3"
