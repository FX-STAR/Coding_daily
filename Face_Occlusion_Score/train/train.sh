#!/bin/bash
nohup ../../build/tools/caffe train -solver solver.prototxt --gpu 0 > log.txt 2>&1 &
