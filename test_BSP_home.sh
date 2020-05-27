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


python3 main.py --gpu $1 --task 'office_home' --optimizer 'Momentum' \
    --target Clipart --partial_domain Realworld Clipart \
    --exp_name "Home_ablation_exponential_RetoCl" \
    --SVD_ld 0.0001 --SVD_norm --SVD_ld_adapt exponential --SVD_k 1 --advcoeff 0.2 --no_MCD

python3 main.py --gpu $1 --task 'office_home' --optimizer 'Momentum' \
    --target Clipart --partial_domain Realworld Clipart \
    --exp_name "Home_ablation_noSVD_RetoCl" \
    --SVD_ld 0. --SVD_norm --SVD_ld_adapt exponential --SVD_k 1 --advcoeff 0.2 --no_MCD
    
python3 main.py --gpu $1 --task 'office_home' --optimizer 'Momentum' \
    --target Product --partial_domain Realworld Product \
    --exp_name "Home_ablation_exponential_RetoPr" \
    --SVD_ld 0.0001 --SVD_norm --SVD_ld_adapt exponential --SVD_k 1 --advcoeff 0.2 --no_MCD

python3 main.py --gpu $1 --task 'office_home' --optimizer 'Momentum' \
    --target Product --partial_domain Realworld Product \
    --exp_name "Home_ablation_noSVD_RetoPr" \
    --SVD_ld 0. --SVD_norm --SVD_ld_adapt exponential --SVD_k 1 --advcoeff 0.2 --no_MCD
