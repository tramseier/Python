#!/bin/bash

mkdir results

for (( c=20; c<=200; c=c+20 ))
do
  ./harmosc_euler $c | grep -v INFO > results/out_euler_$c
  ./harmosc_pc $c | grep -v INFO > results/out_pc_$c
  ./harmosc_verlet $c | grep -v INFO > results/out_verlet_$c
done
