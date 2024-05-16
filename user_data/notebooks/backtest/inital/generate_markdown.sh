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

# Derive the specific output file name based on the time interval
OUTPUT_FILE="${OUTPUT_FILE_PREFIX}_${TIME_INTERVAL}.md"

# Initialize the Markdown file
echo "## 回测概要" > "$OUTPUT_FILE"

# Check if the result directory exists and is not empty
if [ -d "$RESULT_DIR" ] && [ "$(ls -A $RESULT_DIR)" ]; then
    first_file=$(find "$RESULT_DIR" -type f -name "*_${TIME_INTERVAL}.txt" | head -1)
    if [ -n "$first_file" ]; then
        # Extract backtest time range and max open trades
        backtest_range=$(grep -oP 'Backtested \K.*' "$first_file" | head -1)
        max_open_trades=$(grep -oP 'Max open trades : \K\d+' "$first_file" | head -1)

        start_date=$(echo "$backtest_range" | awk '{print $1}')
        end_date=$(echo "$backtest_range" | awk '{print $4}')

        echo "- **回测时间范围**：$start_date 至 $end_date" >> "$OUTPUT_FILE"
        echo "- **最大开放交易数**：$max_open_trades" >> "$OUTPUT_FILE"
    fi
fi

echo "- **周期**：$TIME_INTERVAL" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "| 策略名称 | 交易次数 | 平均利润 % | 累计利润 % | 净利润 USDT | 净利润 % | 平均持续时间 | 胜/平/负 胜率% | 回撤 |" >> "$OUTPUT_FILE"
echo "|----------|--------|-----------|------------|--------------|-----------|-------------|---------------|------|" >> "$OUTPUT_FILE"

for file in $RESULT_DIR/*_${TIME_INTERVAL}.txt; do
    strategy_name=$(basename "$file" .txt | sed "s/_${TIME_INTERVAL}//")

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
    ' "$file" >> "$OUTPUT_FILE"
done

echo "Markdown file $OUTPUT_FILE has been generated."
