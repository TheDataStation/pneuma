#!/bin/bash

prefix="https://uchicago.box.com/shared/static/"

file_ids=(
    "k2lxv036e85b3yuijioc2ql6y0klju9v"
    "ylqm48dfyb3dqw926li26euwib5nutwu"
    "io2yxpmogoaog9dfnigdek2i7y8qbbfq"
    "trnsoveq5d0hy4b3vc2yfvqfpef03k8a"
    "bhd6ujygxtvgbxdwqu4x0fewc3zqfuj4"
    "87p1u0eqg7tgjmxtsiz8ky4aqdpyzc5k"
    "s6o2j8nbzfy46e6jrewq4x89w6dcv485"
)

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for id in "${file_ids[@]}"; do
    full_url="${prefix}${id}"
    echo "Downloading file with ID: $id"
    wget --content-disposition -P "$script_dir" "$full_url"
done
