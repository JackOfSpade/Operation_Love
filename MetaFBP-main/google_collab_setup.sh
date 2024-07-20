#!/bin/bash

# Define the token and repository URL
token='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'

cd /content/drive/MyDrive/ &&
# Clone the repository using the token
git clone https://${token}@github.com/JackOfSpade/Operation_Love.git &&
cd Operation_Love/MetaFBP-main &&
pip install -r requirements.txt

