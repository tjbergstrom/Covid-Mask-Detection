#!/bin/bash
#
# Ty Bergstrom
# auto_train.sh
# CSCE A401
# October 2020
# Software Engineering Project
#
# $ bash auto_train.sh
#
# Build different models automatically and run continuously
# Using for loops to try different combinations of build parameters
# To test out different hypertunings for increased accuracy

# All possible parameters
models=("-m Full_Net " "-m Quick_Net ")
epochs=("-e 25 " "-e 50 " "-e 75 " "-e 100 ")
opt=("-o Adam " "-o Adam2 " "-o Adam3 " "-o Adam4 " "-o SGD " "-o SGD2 " "-o SGD3 " "-o RMSprop " "-o Adadelta ")
aug=("-a light1 " "-a light2 " "-a light3 " "-a medium1 " "-a medium2 " "-a medium3 " "-a heavy1 " "-a heavy2 ")
bs=("-b 16 " "-b 24 " "-b 32 " "-b 42 " "-b 48 " "-b 64" "-b 72 " "-b 128 ")
imgsz=("-i 24 " "-i 32 " "-i 48 " "-i 64 " "-i 72 ")
kernel=("-k 3 " "-k 5")

# Edit the parameters you want for this run

kernel=("-k 3 ")
imgsz=("-i 48 ")
bs=("-b 32 " "-b 48 ")
epochs=("-e 30 ")
opt=("-o Adam1 " "-o Adam2 ")

#source ./venv1/bin/activate

now=`date`
printf "\n\n** Beginning auto_train.sh on $now\n\n" >> performance.txt

cmd="python3 -W ignore train_a_model.py "
model_name="masks_lowQ"
#dataset="../data/masks_hiQ/processed_dataset"
dataset="../data/masks_lowQ/processed_dataset_P"
test_dataset="../data/masks_test"
cmd="python3 -W ignore train_a_model.py -d "$dataset" -s "$model_name
test_cmd="python3 -W ignore ../classification/test_img.py -d "$test_dataset" -m "$model_name
plot="-p plots/plot"
itr=0

# Loop thru kernel sizes
for k in "${!kernel[@]}"
do
    # Loop thru image sizes
    for i in "${!imgsz[@]}"
    do
        # Loop thru batch sizes
        for b in "${!bs[@]}"
        do
            # Loop thru number of epochs
            for e in "${!epochs[@]}"
            do
                # Loop thru optimizers
                for o in "${!opt[@]}"
                do

                    printf "\n $cmd ${models[0]} ${opt[$o]} ${epochs[$e]} ${bs[$b]} ${imgsz[$i]} ${kernel[$k]} ${plot}${itr}.png \n"
                    $cmd ${models[0]} ${opt[$o]} ${epochs[$e]} ${bs[$b]} ${imgsz[$i]} ${kernel[$k]} ${plot}${itr}".png"
                    itr=$((itr+1))
                    #$test_cmd

                done # optimizers
            done # epochs
        done # batch sizes
    done # image sizes
done # kernel sizes

echo Finished auto_train.sh >> performance.txt
# List the 5 builds with the highest accuracy
grep build performance.txt | sort -k 3 -nr | head -5 | awk '{print $2 " " $3}' >> performance.txt
printf "\n\n" >> performance.txt



##
