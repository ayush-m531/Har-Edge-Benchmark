#!/usr/bin/env bash
# Fetches the UCI HAR dataset into the repo root.
set -euo pipefail

URL="https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"

curl -L -o har.zip "$URL"
unzip -q har.zip
unzip -q "UCI HAR Dataset.zip"
rm -f har.zip "UCI HAR Dataset.zip"

echo "Done. Dataset at ./UCI HAR Dataset"
