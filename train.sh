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

domain=("SVHN" "SYNTH" "MNIST")

for ((i=0;i<=0;i++)); do
    python3 main.py --gpu $1 --task 'office' --target Amazon --partial_domain Amazon DSLR \
        --exp_name "office_sourceonly_partial_DtoA_seed"${i}"" \
        --SVD_ld 0. --no_MCD --advcoeff 0.
    python3 main.py --gpu $1 --task 'office' --target Amazon --partial_domain Amazon Webcam \
        --exp_name "office_sourceonly_partial_WtoA_seed"${i}"" \
        --SVD_ld 0. --no_MCD --advcoeff 0.
    python3 main.py --gpu $1 --task 'office' --target Webcam --partial_domain DSLR Webcam \
        --exp_name "office_sourceonly_partial_DtoW_seed"${i}"" \
        --SVD_ld 0. --no_MCD --advcoeff 0.
    python3 main.py --gpu $1 --task 'office' --target DSLR --partial_domain DSLR Webcam \
        --exp_name "office_sourceonly_partial_WtoD_seed"${i}"" \
        --SVD_ld 0. --no_MCD --advcoeff 0.
done

python3 main.py --gpu $1 --task 'office' --target Amazon --partial_domain Amazon DSLR \
    --exp_name "office_sourceonly_partial_DtoA_seed"${i}"" \
            --SVD_ld 0. --no_MCD --advcoeff 0. --num_steps_stop 50000

