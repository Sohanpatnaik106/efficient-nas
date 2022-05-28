# ************************************
# *        DATASET: CIFAR100         *
# ************************************
 
# NOTE: 
#   1. Number of epochs = 180
#   2. VGG19 model

# Experiment 1: Sample an architecture after 50 batches
python ./main.py --batch_norm True --batch_size 128 --criterion_type cross-entropy --data_path ./datasets/cifar100/ 
            --dir_name vgg19 --dropout 0.5 --gpuid 0 --init_weights True --learning_rate 1e-4 \
            --log_dir ./outputs/base_model/num_epochs_180 \
            --model_architecture 64 64 "M" 128 128 "M" 256 256 256 256 "M" 512 512 512 512 "M" 512 512 512 512 "M" \
            --model_config E --model_name vgg19 --num_classes 100 --num_configs 100 --num_epochs 180 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/base_model/num_epochs_180 --weight_decay 1e-4

# NOTE: 
#   1. Number of epochs = 180
#   2. Model 17

# Experiment 1: Sample an architecture after 50 batches
python ./main.py --batch_norm True --batch_size 128 --criterion_type cross-entropy --data_path ./datasets/cifar100/ 
            --dir_name model17 --dropout 0.5 --gpuid 0 --init_weights True --learning_rate 1e-4 \
            --log_dir ./outputs/base_model/num_epochs_180 \
            --model_architecture 64 64 128 "M" 128 256 "M" 256 256 256 "M" 512 512 512 512 512 512 "M" 512 "M" 512 \
            --model_config E --model_name vgg19 --num_classes 100 --num_configs 100 --num_epochs 180 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/base_model/num_epochs_180 --weight_decay 1e-4