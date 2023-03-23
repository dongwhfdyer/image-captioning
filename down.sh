wget -c http://images.cocodataset.org/zips/test2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/zips/train2017.zip

# Path: unzip.sh
unzip test2017.zip
unzip val2017.zip
unzip train2017.zip

# Path: rm.sh
rm test2017.zip
rm val2017.zip
rm train2017.zip

