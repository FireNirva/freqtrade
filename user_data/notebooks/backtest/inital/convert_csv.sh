#!/bin/bash

# Check if a time frame argument was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <time_frame>"
    exit 1
fi

# Use the provided argument for the time frame
timeFrame="$1"

# Define input and output files based on the time frame
inputFile="result_overview_${timeFrame}.md"
outputFile="result_overview_${timeFrame}.csv"

# Check if input file exists
if [ ! -f "$inputFile" ]; then
    echo "Input file $inputFile does not exist."
    exit 2
fi

# Use awk to process the file
awk 'BEGIN {
    FS="|";
    OFS=",";
    print "Strategy Name","Number of Trades","Average Profit %","Total Profit %","Net Profit USDT","Net Profit %","Average Duration","Win/Draw/Loss Win%","Drawdown";
    startProcess = 0;
}
/----------/ {
    startProcess = 1;
    next;
}
startProcess {
    if (NF < 2) {
        next;
    }
    # Convert the day and time duration to a consistent format, e.g., hours
    gsub(/[[:space:]]+/, "", $7);  # Remove spaces
    if ($7 ~ /day/) {
        split($7, a, "day");
        dayPart = a[1] * 24;  # Convert days to hours
        timePart = a[2];
        split(timePart, b, ":");
        $7 = dayPart + b[1] + b[2]/60;  # Convert hours and minutes to decimal hours
    } else {
        split($7, b, ":");
        $7 = b[1] + b[2]/60;  # Convert hours and minutes to decimal hours
    }
    $7 = $7 " hours";  # Append "hours" for clarity

    # Rebuild the record to apply new OFS
    $1 = $1;
    print;
}' "$inputFile" > "$outputFile"

echo "Conversion completed. Output saved to $outputFile"