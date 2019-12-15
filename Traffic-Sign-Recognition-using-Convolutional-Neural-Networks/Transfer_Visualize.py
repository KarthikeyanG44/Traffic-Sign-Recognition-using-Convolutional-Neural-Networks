import os
import shutil


train_dir = r'E:\Gatech Fall 2019\DIP\Final Project\gtsrb-german-traffic-sign\Train'
dest_dir = r'E:\Gatech Fall 2019\DIP\Final Project\gtsrb-german-traffic-sign\Visualize\Test'

for dirs in os.listdir(train_dir):
    fpath = os.listdir(os.path.join(train_dir,dirs))[10]
    print("Copying",fpath)
    src = os.path.join(train_dir,dirs,fpath)
    dest = os.path.join(dest_dir,fpath)
    shutil.copy(src,dest)

print("Copying Done")