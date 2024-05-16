#!/bin/bash

# Directory where the script and strategies_names.txt are located
DIR="$(pwd)"
STRA="all_strategies.txt"

# Directory to store the results
RESULT_DIR="${DIR}/result"

# Global variable for time interval
TIME_INTERVAL="12h"

# Check if the result directory exists, create it if not
[ ! -d "$RESULT_DIR" ] && mkdir -p "$RESULT_DIR"

# Read strategy names from the file and run backtesting for each
while IFS= read -r line; do
    # Trim carriage return and newline just in case
    strategy=$(echo "$line" | tr -d '\r\n')
    
    echo "Running backtesting for strategy: $strategy with time interval: $TIME_INTERVAL"
    # Adjust the command to use the $TIME_INTERVAL variable and include the time frame in the file name
    freqtrade backtesting --config /home/alice/freqtrade/user_data/config.json --userdir /home/alice/freqtrade/user_data --strategy-list "$strategy" --timerange 20181127- -i "$TIME_INTERVAL" > "${RESULT_DIR}/${strategy}_${TIME_INTERVAL}.txt"
    echo "Result stored in ${RESULT_DIR}/${strategy}_${TIME_INTERVAL}.txt"
done < "${DIR}/${STRA}"

echo "Backtesting completed for all strategies." 