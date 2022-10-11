#!/bin/sh

DIM=2;

num_samples=1000;
burn_in=0;

results_path='/home/anne/Masterarbeit/masterarbeit/2d/Example_1/results/samples_';

a=2;
b=2;
c=2;

d=0.01;
e=0.01;
f=0.01;

startPt_x=127
startPt_y=127
startPt_phi=0

mode='MH';

cd /home/anne/Masterarbeit/muq2/build/examples/SamplingAlgorithms/MCMC/MLDA/cpp
./MLDA $DIM $num_samples $burn_in $results_path $a $b $c $d $e $f $startPt_x $startPt_y $startPt_phi $mode