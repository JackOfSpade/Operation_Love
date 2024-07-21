%%bash

# Change directory
cd /content/drive/MyDrive/Operation_Love &&

# Clone the repository using the token
git pull origin main &&

# Change to the repository directory
cd MetaFBP-main &&

# Install required packages
pip install -r requirements.txt

# cd /content/drive/MyDrive/Operation_Love/MetaFBP-main && bash train_fea.sh PFBP-SCUT5500
# cd /content/drive/MyDrive/Operation_Love/MetaFBP-main && bash train_fea.sh PFBP-SCUT500
# cd /content/drive/MyDrive/Operation_Love/MetaFBP-main && bash train_fea.sh PFBP-US10K
# cd /content/drive/MyDrive/Operation_Love/MetaFBP-main && bash train.sh MetaFBP-R PFBP-SCUT5500