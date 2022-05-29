
# ************************************
# *        DATASET: CIFAR100         *
# ************************************

# NOTE: 
#   1. Without exponential moving average of accuracies
#   2. Number of epochs = 180
#   3. Normalisation = False
#   4. Dynamic Normalisation = False

# Experiment 1: 
python ./main.py --architecture_search True --batch_norm True --batch_size 128 \
            --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --gpuid 0 --init_weights True --learning_rate 1e-4 \
            --log_dir ./outputs/epoch_sample/num_epochs_180/no_normalisation/no_mov_avg --model_config E --model_name vgg19 \
            --num_classes 100 --num_configs 100 --num_epochs 180 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/epoch_sample/num_epochs_180/no_normalisation/no_mov_avg --weight_decay 1e-4

# NOTE: 
#   1. With exponential moving average of accuracies
#   2. Number of epochs = 180
#   3. Normalisation = False
#   4. Dynamic Normalisation = False

# Experiment 2: 
python ./main.py --architecture_search True --batch_norm True --batch_size 128 \
            --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --exponential_moving_average True --gpuid 0 --init_weights True \
            --learning_rate 1e-4 --log_dir ./outputs/epoch_sample/num_epochs_180/no_normalisation/mov_avg --model_config E --model_name vgg19 \
            --num_classes 100 --num_configs 100 --num_epochs 180 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/epoch_sample/num_epochs_180/no_normalisation/mov_avg --weight_decay 1e-4

# NOTE: 
#   1. Without exponential moving average of accuracies
#   2. Number of epochs = 220
#   3. Normalisation = False
#   4. Dynamic Normalisation = False

# Experiment 3: 
python ./main.py --architecture_search True --batch_norm True --batch_size 128 \
            --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --gpuid 0 --init_weights True --learning_rate 1e-4 \
            --log_dir ./outputs/epoch_sample/num_epochs_220/no_normalisation/no_mov_avg --model_config E --model_name vgg19 \
            --num_classes 100 --num_configs 100 --num_epochs 220 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/epoch_sample/num_epochs_220/no_normalisation/no_mov_avg --weight_decay 1e-4

# NOTE: 
#   1. With exponential moving average of accuracies
#   2. Number of epochs = 220
#   3. Normalisation = False
#   4. Dynamic Normalisation = False

# Experiment 4: 
python ./main.py --architecture_search True --batch_norm True --batch_size 128 \
            --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --exponential_moving_average True --gpuid 0 --init_weights True \
            --learning_rate 1e-4 --log_dir ./outputs/epoch_sample/num_epochs_220/no_normalisation/mov_avg --model_config E --model_name vgg19 \
            --num_classes 100 --num_configs 100 --num_epochs 220 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/epoch_sample/num_epochs_220/no_normalisation/mov_avg --weight_decay 1e-4

# NOTE: 
#   1. Without exponential moving average of accuracies
#   2. Number of epochs = 250
#   3. Normalisation = False
#   4. Dynamic Normalisation = False

# Experiment 5: 
python ./main.py --architecture_search True --batch_norm True --batch_size 128 \
            --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --gpuid 0 --init_weights True --learning_rate 1e-4 \
            --log_dir ./outputs/epoch_sample/num_epochs_250/no_normalisation/no_mov_avg --model_config E --model_name vgg19 \
            --num_classes 100 --num_configs 100 --num_epochs 250 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/epoch_sample/num_epochs_250/no_normalisation/no_mov_avg --weight_decay 1e-4

# NOTE: 
#   1. With exponential moving average of accuracies
#   2. Number of epochs = 250
#   3. Normalisation = False
#   4. Dynamic Normalisation = False

# Experiment 6: 
python ./main.py --architecture_search True --batch_norm True --batch_size 128 \
            --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --exponential_moving_average True --gpuid 0 --init_weights True \
            --learning_rate 1e-4 --log_dir ./outputs/epoch_sample/num_epochs_250/no_normalisation/mov_avg --model_config E --model_name vgg19 \
            --num_classes 100 --num_configs 100 --num_epochs 250 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/epoch_sample/num_epochs_250/no_normalisation/mov_avg --weight_decay 1e-4

# NOTE: 
#   1. Without exponential moving average of accuracies
#   2. Number of epochs = 180
#   3. Normalisation = True
#   4. Dynamic Normalisation = False

# Experiment 7: 
python ./main.py --architecture_search True --batch_norm True --batch_size 128 \
            --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --gpuid 0 --init_weights True --learning_rate 1e-4 \
            --log_dir ./outputs/epoch_sample/num_epochs_180/normalisation/no_mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 180 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/epoch_sample/num_epochs_180/normalisation/no_mov_avg --weight_decay 1e-4

# NOTE: 
#   1. With exponential moving average of accuracies
#   2. Number of epochs = 180
#   3. Normalisation = True
#   4. Dynamic Normalisation = False

# Experiment 8: 
python ./main.py --architecture_search True --batch_norm True --batch_size 128 \
            --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --exponential_moving_average True --gpuid 0 --init_weights True \
            --learning_rate 1e-4 --log_dir ./outputs/epoch_sample/num_epochs_180/normalisation/mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 180 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/epoch_sample/num_epochs_180/normalisation/mov_avg --weight_decay 1e-4

# NOTE: 
#   1. Without exponential moving average of accuracies
#   2. Number of epochs = 220
#   3. Normalisation = True
#   4. Dynamic Normalisation = False

# Experiment 9: 
python ./main.py --architecture_search True --batch_norm True --batch_size 128 \
            --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --gpuid 0 --init_weights True --learning_rate 1e-4 \
            --log_dir ./outputs/epoch_sample/num_epochs_220/normalisation/no_mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 220 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/epoch_sample/num_epochs_220/normalisation/no_mov_avg --weight_decay 1e-4

# NOTE: 
#   1. With exponential moving average of accuracies
#   2. Number of epochs = 220
#   3. Normalisation = True
#   4. Dynamic Normalisation = False

# Experiment 10: 
python ./main.py --architecture_search True --batch_norm True --batch_size 128 \
            --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --exponential_moving_average True --gpuid 0 --init_weights True \
            --learning_rate 1e-4 --log_dir ./outputs/epoch_sample/num_epochs_220/normalisation/mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 220 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/epoch_sample/num_epochs_220/normalisation/mov_avg --weight_decay 1e-4

# NOTE: 
#   1. Without exponential moving average of accuracies
#   2. Number of epochs = 250
#   3. Normalisation = True
#   4. Dynamic Normalisation = False

# Experiment 11: 
python ./main.py --architecture_search True --batch_norm True --batch_size 128 \
            --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --gpuid 0 --init_weights True --learning_rate 1e-4 \
            --log_dir ./outputs/epoch_sample/num_epochs_250/normalisation/no_mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 250 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/epoch_sample/num_epochs_250/normalisation/no_mov_avg --weight_decay 1e-4

# NOTE: 
#   1. With exponential moving average of accuracies
#   2. Number of epochs = 250
#   3. Normalisation = True
#   4. Dynamic Normalisation = False

# Experiment 12: 
python ./main.py --architecture_search True --batch_norm True --batch_size 128 \
            --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --exponential_moving_average True --gpuid 0 --init_weights True \
            --learning_rate 1e-4 --log_dir ./outputs/epoch_sample/num_epochs_250/normalisation/mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 250 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/epoch_sample/num_epochs_250/normalisation/mov_avg --weight_decay 1e-4

# NOTE: 
#   1. Without exponential moving average of accuracies
#   2. Number of epochs = 180
#   3. Normalisation = True
#   4. Dynamic Normalisation = True

# Experiment 13: 
python ./main.py --architecture_search True --batch_norm True --batch_size 128 \
            --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 dynamic_temperature True --gpuid 0 --init_weights True --learning_rate 1e-4 \
            --log_dir ./outputs/epoch_sample/num_epochs_180/dynamic_normalisation/no_mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 180 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 10 --temperature_epoch_scaling 50 --track_running_stats True \
            --visualisation_dir ./visualisation/epoch_sample/num_epochs_180/dynamic_normalisation/no_mov_avg --weight_decay 1e-4

# NOTE: 
#   1. With exponential moving average of accuracies
#   2. Number of epochs = 180
#   3. Normalisation = True
#   4. Dynamic Normalisation = True

# Experiment 14: 
python ./main.py --architecture_search True --batch_norm True --batch_size 128 \
            --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 dynamic_temperature True --exponential_moving_average True --gpuid 0 --init_weights True \
            --learning_rate 1e-4 --log_dir ./outputs/epoch_sample/num_epochs_180/dynamic_normalisation/mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 180 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 10 --temperature_epoch_scaling 50 --track_running_stats True \
            --visualisation_dir ./visualisation/epoch_sample/num_epochs_180/dynamic_normalisation/mov_avg --weight_decay 1e-4

# NOTE: 
#   1. Without exponential moving average of accuracies
#   2. Number of epochs = 220
#   3. Normalisation = True
#   4. Dynamic Normalisation = True

# Experiment 15: 
python ./main.py --architecture_search True --batch_norm True --batch_size 128 \
            --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 dynamic_temperature True --gpuid 0 --init_weights True --learning_rate 1e-4 \
            --log_dir ./outputs/epoch_sample/num_epochs_220/dynamic_normalisation/no_mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 220 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 10 --temperature_epoch_scaling 50 --track_running_stats True \
            --visualisation_dir ./visualisation/epoch_sample/num_epochs_220/dynamic_normalisation/no_mov_avg --weight_decay 1e-4

# NOTE: 
#   1. With exponential moving average of accuracies
#   2. Number of epochs = 220
#   3. Normalisation = True
#   4. Dynamic Normalisation = True

# Experiment 16: 
python ./main.py --architecture_search True --batch_norm True --batch_size 128 \
            --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 dynamic_temperature True --exponential_moving_average True --gpuid 0 --init_weights True \
            --learning_rate 1e-4 --log_dir ./outputs/epoch_sample/num_epochs_220/dynamic_normalisation/mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 220 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 10 --temperature_epoch_scaling 50 --track_running_stats True \
            --visualisation_dir ./visualisation/epoch_sample/num_epochs_220/dynamic_normalisation/mov_avg --weight_decay 1e-4

# NOTE: 
#   1. Without exponential moving average of accuracies
#   2. Number of epochs = 250
#   3. Normalisation = True
#   4. Dynamic Normalisation = True

# Experiment 17: 
python ./main.py --architecture_search True --batch_norm True --batch_size 128 \
            --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 dynamic_temperature True --gpuid 0 --init_weights True --learning_rate 1e-4 \
            --log_dir ./outputs/epoch_sample/num_epochs_250/dynamic_normalisation/no_mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 250 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 10 --temperature_epoch_scaling 50 --track_running_stats True \
            --visualisation_dir ./visualisation/epoch_sample/num_epochs_250/dynamic_normalisation/no_mov_avg --weight_decay 1e-4

# NOTE: 
#   1. With exponential moving average of accuracies
#   2. Number of epochs = 250
#   3. Normalisation = True
#   4. Dynamic Normalisation = True

# Experiment 18: 
python ./main.py --architecture_search True --batch_norm True --batch_size 128 \
            --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 dynamic_temperature True --exponential_moving_average True --gpuid 0 --init_weights True \
            --learning_rate 1e-4 --log_dir ./outputs/epoch_sample/num_epochs_250/dynamic_normalisation/mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 250 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 10 --temperature_epoch_scaling 50 --track_running_stats True \
            --visualisation_dir ./visualisation/epoch_sample/num_epochs_250/dynamic_normalisation/mov_avg --weight_decay 1e-4