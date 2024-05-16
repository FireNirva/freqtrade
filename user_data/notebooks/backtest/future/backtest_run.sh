#!/bin/bash

# Directory where the script and picked_strategies.txt are located
DIR="$(pwd)"
STRA="picked_strategies.txt"

# Directory to store the results
RESULT_DIR="${DIR}/result_fee"

# Check if the result directory exists, create it if not
[ ! -d "$RESULT_DIR" ] && mkdir -p "$RESULT_DIR"

# Define time intervals
TIME_INTERVALS=("5m" "15m" "1h" "4h")

# Define market phases with their respective time ranges
declare -A MARKET_PHASES
MARKET_PHASES=(
    ["uptrend"]="20200901-20210320"
    ["sideways"]="20210409-20211027"
    ["downtrend"]="20211113-20220601"
)

# Define trading fee
TRADING_FEE="0.00075"

# Read strategy names from the file and run backtesting for each
while IFS= read -r line; do
    # Trim carriage return and newline just in case
    strategy=$(echo "$line" | tr -d '\r\n')

    for phase in "${!MARKET_PHASES[@]}"; do
        for interval in "${TIME_INTERVALS[@]}"; do
            echo "Running backtesting for strategy: $strategy with time interval: $interval during $phase phase"
            # Adjust the command to use the $interval and $TRADING_FEE variables, and include the time frame and phase in the file name
            freqtrade backtesting --config "${DIR}/config.json" --userdir /home/alice/freqtrade/user_data --strategy-list "$strategy" --timerange "${MARKET_PHASES[$phase]}" -i "$interval" --fee "$TRADING_FEE" > "${RESULT_DIR}/${strategy}_${phase}_${interval}.txt"
            echo "Result stored in ${RESULT_DIR}/${strategy}_${phase}_${interval}.txt"
        done
    done
done < "${DIR}/${STRA}"

echo "Backtesting completed for all strategies."
