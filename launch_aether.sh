#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
source aether_env/bin/activate 2>/dev/null || source aether_env/Scripts/activate 2>/dev/null
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs) 2>/dev/null
fi
python3 main.py --mode agent "$@"
