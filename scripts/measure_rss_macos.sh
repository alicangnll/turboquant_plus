#!/bin/bash
# Run llama-cli under macOS /usr/bin/time -l and read "maximum resident set size" (peak RSS).
# Both llama and time write to stderr; scroll to the bottom for the RSS line.
#
# Usage:
#   ./scripts/measure_rss_macos.sh /path/to/llama-cli [llama-cli args...]
# Example:
#   chmod +x scripts/measure_rss_macos.sh
#   ./scripts/measure_rss_macos.sh ./llama-cpp-turboquant/build/bin/llama-cli -m models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf -p "test" -n 2 -ngl 20

set -euo pipefail

if [[ "$OSTYPE" != darwin* ]]; then
  echo "This script expects macOS (/usr/bin/time -l)." >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/llama-cli [args...]" >&2
  exit 1
fi

if [[ ! -f "$1" ]]; then
  echo "Not found: $1" >&2
  exit 1
fi

echo ">>> Tip: compare runs with different -c / --batch-size / -ngl / cache types."
echo ">>> At end of output, find: maximum resident set size (bytes)"
echo ""

exec /usr/bin/time -l "$@"
