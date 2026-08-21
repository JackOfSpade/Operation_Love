#!/bin/sh
set -eu

cd /content/drive/MyDrive/
repo_url=${OPERATION_LOVE_REPO_URL:-https://github.com/JackOfSpade/Operation_Love.git}
git clone "$repo_url"
