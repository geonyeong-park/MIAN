#!/bin/bash

<< "END"
advcoeff=(0.1 0.05 0.01 0.005)
DGlr=(0.0003 0.0005)
pixAdv='LS'
Gfake_cyc=(0.1 1)

for lr in "${DGlr[@]}"; do
    for Gc in "${Gfake_cyc[@]}"; do
        for ac in "${advcoeff[@]}"; do
            python3 main.py --gpu 0 1 --advcoeff $ac --Gfake_cyc $Gc --DGlr $lr --pixAdv $pixAdv\
                --exp_name "office_Amazon_lr_"${lr}"_Gc_"${Gc}"_ac_"${ac}""
        done
    done
done

END

<<<<<<< 2b4784aa5663325d01434f29dba2199e9269963b
domain=("Amazon" "Caltech" "DSLR" "Webcam")
lambda=(1.0 0.8 1.2 0.6 1.4)
||||||| merged common ancestors
domain=("Caltech" "Amazon" "DSLR" "Webcam")
lambda=(1. 1.2 0.8 1.6 0.6)
=======
domain=("Caltech" "Amazon" "DSLR" "Webcam")
>>>>>>> testing with Adam opt

for ((i=0;i<=3;i++))
do
<<<<<<< 2b4784aa5663325d01434f29dba2199e9269963b
    for target in "${domain[@]}"; do
        python3 main.py --gpu $1 --task 'office_caltech_10' --optimizer 'Adam' --target $target \
            --exp_name "office_DAdam_"${target}"_ld_0.1_seed_"${i}"" --advcoeff 0.1
||||||| merged common ancestors
    for ld in "${lambda[@]}"; do
        for target in "${domain[@]}"; do
            python3 main.py --gpu $1 --task 'office_caltech_10' --optimizer 'Momentum' --target $target \
                --exp_name "office_DAdam_"${target}"_ld_"${ld}"_seed_"${i}"" --advcoeff $ld
        done
=======
    for target in "${domain[@]}"; do
        python3 main.py --gpu $1 --task 'office_caltech_10' --optimizer 'Adam' --target $target \
            --exp_name "office_MCD_merged_"${target}"_seed_"${i}"" --advcoeff 0. 
>>>>>>> testing with Adam opt
    done
done
