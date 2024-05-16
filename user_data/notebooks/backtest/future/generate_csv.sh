#!/bin/bash

# Check if the user has provided an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <time_interval>"
    exit 1
fi

# Use the first argument as the time interval
TIME_INTERVAL="$1"
OUTPUT_FILE_PREFIX="result_overview"
RESULT_DIR="result_fee" # Common directory for all time frames

# Create the output directory if it does not exist
mkdir -p "$OUTPUT_FILE_PREFIX"

# Function to generate the CSV file for a specific trend
generate_csv() {
    local trend=$1
    local output_file="${OUTPUT_FILE_PREFIX}/${OUTPUT_FILE_PREFIX}_${trend}_${TIME_INTERVAL}.csv"

    # Initialize the CSV file
    echo "Strategy Name,Number of Trades,Average Profit %,Total Profit %,Net Profit USDT,Net Profit %,Average Duration,Win/Draw/Loss Win%,Drawdown,Sharpe" > "$output_file"

    for file in $RESULT_DIR/*_${trend}_${TIME_INTERVAL}.txt; do
        strategy_name=$(basename "$file" .txt | sed "s/_${trend}_${TIME_INTERVAL}//")
        
        echo "Processing file: $file" # Debug info

        sharpe_ratio=$(awk '/Sharpe/ {print $4}' "$file")

        awk -v strategy="$strategy_name" -v trend="$trend" -v sharpe="$sharpe_ratio" '
        BEGIN {FS = "|"; OFS = "|"; count = 0;}
        $0 ~ strategy {
            count++;
            if (count == 2) {
                for(i = 2; i <= NF; i++) {
                    gsub(/^[ \t]+|[ \t]+$/, "", $i);
                }
                gsub(/,/, "", $8);  # Remove commas from $8
                row = strategy "," $3 "," $4 "," $5 "," $6 "," $7 "," $8 "," $9 "," $10 "," sharpe;
                print row;
                count = 0;
            }
        }
        ' "$file" >> "$output_file"
    done

    echo "CSV file $output_file has been generated."
}

# Generate CSV files for each trend
for trend in uptrend sideways downtrend; do
    generate_csv $trend
done
