#!/bin/bash
SRC='/mnt/c/Users/qianc/Projects/yolo_Blood/src/blood_label/raw'
DST='/mnt/g/WSL/calib_images'

echo '=== origin 图数量 ==='
ls "$SRC"/*.origin.jpg 2>/dev/null | wc -l

echo '=== 清空旧校准图，复制全部 origin 图 ==='
rm -f "$DST"/*.jpg
cp "$SRC"/*.origin.jpg "$DST"/

echo '校准图数量:'
ls "$DST"/*.jpg | wc -l

echo '=== 生成 calib.txt ==='
ls "$DST"/*.jpg > "$DST"/calib.txt
echo 'calib.txt 行数:'
wc -l < "$DST"/calib.txt
