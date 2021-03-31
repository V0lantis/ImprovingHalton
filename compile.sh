#!/bin/bash

set -e
set -x

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $DIR/venv/bin/activate
cython -a Halton.pyx
pip install -e .
open -a safari Halton.html
