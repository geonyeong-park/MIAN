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

for ((i=1;i<=3;i++)); do
    python3 main.py --gpu $1 --task 'office_home' --target Art --partial_domain Art Clipart \
        --exp_name "Home_sourceonly_partial_CltoAr_seed"${i}"" \
        --SVD_ld 0. --no_MCD --advcoeff 0.
    python3 main.py --gpu $1 --task 'office_home' --target Clipart --partial_domain Art Clipart \
        --exp_name "Home_sourceonly_partial_ArtoCl_seed"${i}"" \
        --SVD_ld 0. --no_MCD --advcoeff 0.
    python3 main.py --gpu $1 --task 'office_home' --target Product --partial_domain Product Real \
        --exp_name "Home_sourceonly_partial_RtoP_seed"${i}"" \
        --SVD_ld 0. --no_MCD --advcoeff 0.
    python3 main.py --gpu $1 --task 'office_home' --target Real --partial_domain Product Real \
        --exp_name "Home_sourceonly_partial_PtoR_seed"${i}"" \
        --SVD_ld 0. --no_MCD --advcoeff 0.
done

python3 main.py --gpu $1 --task 'office_home' --target Real --partial_domain Product Real \
    --exp_name "Home_sourceonly_partial_PtoR_seed0" \
    --SVD_ld 0. --no_MCD --advcoeff 0.
