#!/bin/bash

# Define the directory containing the strategies
STRATEGIES_DIR="/home/alice/freqtrade/user_data/strategies"

# Define the output file path
OUTPUT_FILE="/home/alice/freqtrade/user_data/notebooks/all_strategies.txt"

# Navigate to the strategies directory
cd "$STRATEGIES_DIR"

# Check if the directory exists and is accessible
if [ $? -eq 0 ]; then
    # List all .py files, remove the '.py' extension, and write the results to the output file
    ls *.py | sed 's/\.py$//' > "$OUTPUT_FILE"

    echo "All strategy names have been written to $OUTPUT_FILE."
else
    echo "Error: Could not access the directory $STRATEGIES_DIR."
fi
