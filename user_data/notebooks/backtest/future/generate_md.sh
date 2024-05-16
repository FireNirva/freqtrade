#!/bin/bash

# Check if the user has provided an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <time_interval>"
    exit 1
fi

# Use the first argument as the time interval
TIME_INTERVAL="$1"
OUTPUT_FILE_PREFIX="result_overview"
RESULT_DIR="result" # Common directory for all time frames

# Function to generate the markdown file for a specific trend
generate_markdown() {
    local trend=$1
    local output_file="${OUTPUT_FILE_PREFIX}_${trend}_${TIME_INTERVAL}.md"

    # Initialize the Markdown file
    echo "## 回测概要" > "$output_file"

    # Check if the result directory exists and is not empty
    if [ -d "$RESULT_DIR" ] && [ "$(ls -A $RESULT_DIR)" ]; then
        first_file=$(find "$RESULT_DIR" -type f -name "*_${trend}_${TIME_INTERVAL}.txt" | head -1)
        if [ -n "$first_file" ]; then
            # Extract backtest time range and max open trades
            backtest_range=$(grep -oP 'Backtested \K.*' "$first_file" | head -1)
            max_open_trades=$(grep -oP 'Max open trades : \K\d+' "$first_file" | head -1)

            start_date=$(echo "$backtest_range" | awk '{print $1}')
            end_date=$(echo "$backtest_range" | awk '{print $4}')

            echo "- **回测时间范围**：$start_date 至 $end_date" >> "$output_file"
            echo "- **最大开放交易数**：$max_open_trades" >> "$output_file"
        fi
    fi

    echo "- **周期**：$TIME_INTERVAL" >> "$output_file"
    echo "" >> "$output_file"
    echo "| 策略名称 | 交易次数 | 平均利润 % | 累计利润 % | 净利润 USDT | 净利润 % | 平均持续时间 | 胜/平/负 胜率% | 回撤 |" >> "$output_file"
    echo "|----------|--------|-----------|------------|--------------|-----------|-------------|---------------|------|" >> "$output_file"

    for file in $RESULT_DIR/*_${trend}_${TIME_INTERVAL}.txt; do
        strategy_name=$(basename "$file" .txt | sed "s/_${trend}_${TIME_INTERVAL}//")

        awk -v strategy="$strategy_name" '
        BEGIN {FS = "|"; OFS = "|"; count = 0;}
        $0 ~ strategy {
            count++;
            if (count == 2) {
                for(i = 2; i <= NF; i++) {
                    gsub(/^[ \t]+|[ \t]+$/, "", $i);
                }
                print strategy, $3, $4, $5, $6, $7, $8, $9, $10;
                count = 0;
            }
        }
        ' "$file" >> "$output_file"
    done

    echo "Markdown file $output_file has been generated."
}

# Generate markdown files for each trend
for trend in uptrend sideways downtrend; do
    generate_markdown $trend
done
