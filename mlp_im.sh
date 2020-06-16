#!/bin/bash

models=('resnet18' 'resnet152' 'vgg11')
lr_assoc_vals=(0.2 0.3 0.4 0.5)
lr_attn_vals=(0.0013 0.0023 0.0033 0.0043)
phi_vals=(2.0 2.5)
datasets=('shj_images_set1')
losses=('hinge' 'humble' 'mse' 'll')
epochs=128

for model in "${models[@]}"
do
	for lr_assoc in "${lr_assoc_vals[@]}"
	do
		for lr_attn in "${lr_attn_vals[@]}"
		do
			for c in "${c_vals[@]}"
			do
				for phi in "${phi_vals[@]}"
				do
					for dataset in "${datasets[@]}"
					do
						for loss in "${losses[@]}"
						do
							python alcove.py -m 'mlp' -n $model --lr_assoc $lr_assoc --lr_attn $lr_attn --phi $phi -d $dataset -l $loss -e $epochs
						done
					done
				done
			done
		done
	done
done