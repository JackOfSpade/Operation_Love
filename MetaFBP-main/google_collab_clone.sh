%%bash

# Define the token and repository URL
token='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'

# Change directory
cd /content/drive/MyDrive/ &&

# Clone the repository using the token
git clone https://${token}@github.com/JackOfSpade/Operation_Love.git &&

# Change to the repository directory
cd Operation_Love/MetaFBP-main &&

# Install required packages
pip install -r requirements.txt
