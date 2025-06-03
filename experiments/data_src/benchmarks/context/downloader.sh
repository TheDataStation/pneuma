#!/bin/bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
prefix="https://uchicago.box.com/shared/static/"
full_url="${prefix}8zzjmmqflp89kipu0vr7hz6pk9o4ypye"
wget --content-disposition -P "$script_dir" "$full_url"
zip_file=$(find "$script_dir" -maxdepth 1 -name "*.zip" -print -quit)

if [[ -f "$zip_file" ]]; then
    unzip "$zip_file" -d "$script_dir"
    rm "$zip_file"
else
    echo "No ZIP file found. Something went wrong."
fi
