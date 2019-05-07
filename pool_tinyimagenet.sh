#!/bin/bash

wd=~/Colorful-Colarization-Project
dest=$wd/tiny-imagenet-200/all_train/

mkdir $dest

echo "Running pool_tinyimagenet.sh:"
for directory in $wd/tiny-imagenet-200/train/*/
do
  echo $directory
  dir=`echo $directory"images/"`
  files=`ls $dir`
  cd $dir
  cp $files $dest
done
