#!/usr/bin/env bash
set -euo pipefail

smac_commit=0de603e6a67d867b3b13582dfa5aebf36cab7f96
smac_archive_sha256=8e1d782ec3b6347111ff8570a448a4c8fa902d69c0e0bd6fcac56b37b90491eb
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
thirdparty_dir="$repo_root/pymarl-master/3rdparty"
smac_dir="$thirdparty_dir/smac-$smac_commit"
smac_archive="$thirdparty_dir/smac-$smac_commit.tar.gz"

unset LD_LIBRARY_PATH PYTHONPATH
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy

if [[ ${CONDA_DEFAULT_ENV:-} != "macc" ]]; then
    echo "Activate the macc Conda environment before running this script." >&2
    exit 1
fi

mkdir -p "$thirdparty_dir"
if [[ ! -f "$smac_archive" ]]; then
    curl -L --fail --retry 10 --retry-all-errors \
        --connect-timeout 20 --max-time 600 \
        -o "$smac_archive.part" \
        "https://codeload.github.com/oxwhirl/smac/tar.gz/$smac_commit"
    mv "$smac_archive.part" "$smac_archive"
fi

echo "$smac_archive_sha256  $smac_archive" | sha256sum --check --status

if [[ ! -d "$smac_dir/smac" ]]; then
    tar -xzf "$smac_archive" -C "$thirdparty_dir"
fi

python -m pip install --no-deps --no-build-isolation "$smac_dir"

echo "Installed SMAC from commit archive $smac_commit"
