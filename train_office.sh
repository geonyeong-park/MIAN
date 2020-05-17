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

domain=("Amazon" "DSLR" "Webcam")
k=(1 2 4 5 6 7 8)

for ((i=0;i<=3;i++)); do
    for target in "${domain[@]}"; do
        python3 main.py --gpu $1 --task 'office' --optimizer 'Momentum' --target $target \
            --exp_name "adv0.05_ld1e-3_"${target}"_seed_"${i}"" \
            --SVD_ld 0.001 --advcoeff 0.05
    done
done
