
# ************************************
# *        DATASET: CIFAR100         *
# ************************************
 
# NOTE: 
#   1. Without exponential moving average of accuracies
#   2. Number of epochs = 180
#   3. Normalisation = False
#   4. Dynamic Normalisation = False

# Experiment 1: Sample an architecture after 50 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 50 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --gpuid 0 --init_weights True --learning_rate 1e-4 \
            --log_dir ./outputs/batch_sample/sample_50/num_epochs_180/no_normalisation/no_mov_avg --model_config E --model_name vgg19 \
            --num_classes 100 --num_configs 100 --num_epochs 180 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_50/num_epochs_180/no_normalisation/no_mov_avg --weight_decay 1e-4

# Experiment 2: Sample an architecture after 100 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 100 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --gpuid 0 --init_weights True --learning_rate 1e-4 \
            --log_dir ./outputs/batch_sample/sample_100/num_epochs_180/no_normalisation/no_mov_avg --model_config E --model_name vgg19 \
            --num_classes 100 --num_configs 100 --num_epochs 180 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_100/num_epochs_180/no_normalisation/no_mov_avg --weight_decay 1e-4

# NOTE: 
#   1. With exponential moving average of accuracies
#   2. Number of epochs = 180
#   3. Normalisation = False
#   4. Dynamic Normalisation = False

# Experiment 3: Sample an architecture after 50 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 50 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --exponential_moving_average True --gpuid 0 --init_weights True \
            --learning_rate 1e-4 --log_dir ./outputs/batch_sample/sample_50/num_epochs_180/no_normalisation/mov_avg --model_config E --model_name vgg19 \
            --num_classes 100 --num_configs 100 --num_epochs 180 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_50/num_epochs_180/no_normalisation/mov_avg --weight_decay 1e-4

# Experiment 4: Sample an architecture after 100 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 100 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --exponential_moving_average True --gpuid 0 --init_weights True \
            --learning_rate 1e-4 --log_dir ./outputs/batch_sample/sample_100/num_epochs_180/no_normalisation/mov_avg --model_config E --model_name vgg19 \
            --num_classes 100 --num_configs 100 --num_epochs 180 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_100/num_epochs_180/no_normalisation/mov_avg --weight_decay 1e-4

# NOTE: 
#   1. Without exponential moving average of accuracies
#   2. Number of epochs = 220
#   3. Normalisation = False
#   4. Dynamic Normalisation = False

# Experiment 5: Sample an architecture after 50 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 50 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --gpuid 0 --init_weights True --learning_rate 1e-4 \
            --log_dir ./outputs/batch_sample/sample_50/num_epochs_220/no_normalisation/no_mov_avg --model_config E --model_name vgg19 \
            --num_classes 100 --num_configs 100 --num_epochs 220 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_50/num_epochs_220/no_normalisation/no_mov_avg --weight_decay 1e-4

# Experiment 6: Sample an architecture after 100 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 100 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --gpuid 0 --init_weights True --learning_rate 1e-4 \
            --log_dir ./outputs/batch_sample/sample_100/num_epochs_220/no_normalisation/no_mov_avg --model_config E --model_name vgg19 \
            --num_classes 100 --num_configs 100 --num_epochs 220 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_100/num_epochs_220/no_normalisation/no_mov_avg --weight_decay 1e-4

# NOTE: 
#   1. With exponential moving average of accuracies
#   2. Number of epochs = 220
#   3. Normalisation = False
#   4. Dynamic Normalisation = False

# Experiment 7: Sample an architecture after 50 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 50 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --exponential_moving_average True --gpuid 0 --init_weights True \
            --learning_rate 1e-4 --log_dir ./outputs/batch_sample/sample_50/num_epochs_220/no_normalisation/mov_avg --model_config E --model_name vgg19 \
            --num_classes 100 --num_configs 100 --num_epochs 220 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_50/num_epochs_220/no_normalisation/mov_avg --weight_decay 1e-4

# Experiment 8: Sample an architecture after 100 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 100 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --exponential_moving_average True --gpuid 0 --init_weights True \
            --learning_rate 1e-4 --log_dir ./outputs/batch_sample/sample_100/num_epochs_220/no_normalisation/mov_avg --model_config E --model_name vgg19 \
            --num_classes 100 --num_configs 100 --num_epochs 220 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_100/num_epochs_220/no_normalisation/mov_avg --weight_decay 1e-4

# NOTE: 
#   1. Without exponential moving average of accuracies
#   2. Number of epochs = 250
#   3. Normalisation = False
#   4. Dynamic Normalisation = False

# Experiment 9: Sample an architecture after 50 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 50 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --gpuid 0 --init_weights True --learning_rate 1e-4 \
            --log_dir ./outputs/batch_sample/sample_50/num_epochs_250/no_normalisation/no_mov_avg --model_config E --model_name vgg19 \
            --num_classes 100 --num_configs 100 --num_epochs 250 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_50/num_epochs_250/no_normalisation/no_mov_avg --weight_decay 1e-4

# Experiment 10: Sample an architecture after 100 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 100 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --gpuid 0 --init_weights True --learning_rate 1e-4 \
            --log_dir ./outputs/batch_sample/sample_100/num_epochs_250/no_normalisation/no_mov_avg --model_config E --model_name vgg19 \
            --num_classes 100 --num_configs 100 --num_epochs 250 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_100/num_epochs_250/no_normalisation/no_mov_avg --weight_decay 1e-4

# NOTE: 
#   1. With exponential moving average of accuracies
#   2. Number of epochs = 250
#   3. Normalisation = False
#   4. Dynamic Normalisation = False

# Experiment 11: Sample an architecture after 50 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 50 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --exponential_moving_average True --gpuid 0 --init_weights True \
            --learning_rate 1e-4 --log_dir ./outputs/batch_sample/sample_50/num_epochs_250/no_normalisation/mov_avg --model_config E --model_name vgg19 \
            --num_classes 100 --num_configs 100 --num_epochs 250 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_50/num_epochs_250/no_normalisation/mov_avg --weight_decay 1e-4

# Experiment 12: Sample an architecture after 100 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 100 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --exponential_moving_average True --gpuid 0 --init_weights True \
            --learning_rate 1e-4 --log_dir ./outputs/batch_sample/sample_100/num_epochs_250/no_normalisation/mov_avg --model_config E --model_name vgg19 \
            --num_classes 100 --num_configs 100 --num_epochs 250 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_100/num_epochs_250/no_normalisation/mov_avg --weight_decay 1e-4

# NOTE: 
#   1. Without exponential moving average of accuracies
#   2. Number of epochs = 180
#   3. Normalisation = True
#   4. Dynamic Normalisation = False

# Experiment 13: Sample an architecture after 50 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 50 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --gpuid 0 --init_weights True --learning_rate 1e-4 \
            --log_dir ./outputs/batch_sample/sample_50/num_epochs_180/normalisation/no_mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 180 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_50/num_epochs_180/normalisation/no_mov_avg --weight_decay 1e-4

# Experiment 14: Sample an architecture after 100 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 100 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --gpuid 0 --init_weights True --learning_rate 1e-4 \
            --log_dir ./outputs/batch_sample/sample_100/num_epochs_180/normalisation/no_mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 180 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_100/num_epochs_180/normalisation/no_mov_avg --weight_decay 1e-4

# NOTE: 
#   1. With exponential moving average of accuracies
#   2. Number of epochs = 180
#   3. Normalisation = True
#   4. Dynamic Normalisation = False

# Experiment 15: Sample an architecture after 50 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 50 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --exponential_moving_average True --gpuid 0 --init_weights True \
            --learning_rate 1e-4 --log_dir ./outputs/batch_sample/sample_50/num_epochs_180/normalisation/mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 180 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_50/num_epochs_180/normalisation/mov_avg --weight_decay 1e-4

# Experiment 16: Sample an architecture after 100 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 100 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --exponential_moving_average True --gpuid 0 --init_weights True \
            --learning_rate 1e-4 --log_dir ./outputs/batch_sample/sample_100/num_epochs_180/normalisation/mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 180 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_100/num_epochs_180/normalisation/mov_avg --weight_decay 1e-4

# NOTE: 
#   1. Without exponential moving average of accuracies
#   2. Number of epochs = 220
#   3. Normalisation = True
#   4. Dynamic Normalisation = False

# Experiment 17: Sample an architecture after 50 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 50 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --gpuid 0 --init_weights True --learning_rate 1e-4 \
            --log_dir ./outputs/batch_sample/sample_50/num_epochs_220/normalisation/no_mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 220 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_50/num_epochs_220/normalisation/no_mov_avg --weight_decay 1e-4

# Experiment 18: Sample an architecture after 100 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 100 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --gpuid 0 --init_weights True --learning_rate 1e-4 \
            --log_dir ./outputs/batch_sample/sample_100/num_epochs_220/normalisation/no_mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 220 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_100/num_epochs_220/normalisation/no_mov_avg --weight_decay 1e-4

# NOTE: 
#   1. With exponential moving average of accuracies
#   2. Number of epochs = 220
#   3. Normalisation = True
#   4. Dynamic Normalisation = False

# Experiment 19: Sample an architecture after 50 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 50 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --exponential_moving_average True --gpuid 0 --init_weights True \
            --learning_rate 1e-4 --log_dir ./outputs/batch_sample/sample_50/num_epochs_220/normalisation/mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 220 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_50/num_epochs_220/normalisation/mov_avg --weight_decay 1e-4

# Experiment 20: Sample an architecture after 100 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 100 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --exponential_moving_average True --gpuid 0 --init_weights True \
            --learning_rate 1e-4 --log_dir ./outputs/batch_sample/sample_100/num_epochs_220/normalisation/mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 220 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_100/num_epochs_220/normalisation/mov_avg --weight_decay 1e-4

# NOTE: 
#   1. Without exponential moving average of accuracies
#   2. Number of epochs = 250
#   3. Normalisation = True
#   4. Dynamic Normalisation = False

# Experiment 21: Sample an architecture after 50 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 50 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --gpuid 0 --init_weights True --learning_rate 1e-4 \
            --log_dir ./outputs/batch_sample/sample_50/num_epochs_250/normalisation/no_mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 250 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_50/num_epochs_250/normalisation/no_mov_avg --weight_decay 1e-4

# Experiment 22: Sample an architecture after 100 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 100 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --gpuid 0 --init_weights True --learning_rate 1e-4 \
            --log_dir ./outputs/batch_sample/sample_100/num_epochs_250/normalisation/no_mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 250 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_100/num_epochs_250/normalisation/no_mov_avg --weight_decay 1e-4

# NOTE: 
#   1. With exponential moving average of accuracies
#   2. Number of epochs = 250
#   3. Normalisation = True
#   4. Dynamic Normalisation = False

# Experiment 23: Sample an architecture after 50 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 50 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --exponential_moving_average True --gpuid 0 --init_weights True \
            --learning_rate 1e-4 --log_dir ./outputs/batch_sample/sample_50/num_epochs_250/normalisation/mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 250 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_50/num_epochs_250/normalisation/mov_avg --weight_decay 1e-4

# Experiment 24: Sample an architecture after 100 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 100 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 --exponential_moving_average True --gpuid 0 --init_weights True \
            --learning_rate 1e-4 --log_dir ./outputs/batch_sample/sample_100/num_epochs_250/normalisation/mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 250 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_tydynamic Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 0.7 --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_100/num_epochs_250/normalisation/mov_avg --weight_decay 1e-4

# NOTE: 
#   1. Without exponential moving average of accuracies
#   2. Number of epochs = 180
#   3. Normalisation = True
#   4. Dynamic Normalisation = True

# Experiment 25: Sample an architecture after 50 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 50 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 dynamic_temperature True --gpuid 0 --init_weights True --learning_rate 1e-4 \
            --log_dir ./outputs/batch_sample/sample_50/num_epochs_180/dynamic_normalisation/no_mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 180 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 10 --temperature_epoch_scaling True --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_50/num_epochs_180/dynamic_normalisation/no_mov_avg --weight_decay 1e-4

# Experiment 26: Sample an architecture after 100 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 100 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 dynamic_temperature True --gpuid 0 --init_weights True --learning_rate 1e-4 \
            --log_dir ./outputs/batch_sample/sample_100/num_epochs_180/dynamic_normalisation/no_mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 180 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 10 --temperature_epoch_scaling True --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_100/num_epochs_180/dynamic_normalisation/no_mov_avg --weight_decay 1e-4

# NOTE: 
#   1. With exponential moving average of accuracies
#   2. Number of epochs = 180
#   3. Normalisation = True
#   4. Dynamic Normalisation = True

# Experiment 27: Sample an architecture after 50 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 50 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 dynamic_temperature True --exponential_moving_average True --gpuid 0 --init_weights True \
            --learning_rate 1e-4 --log_dir ./outputs/batch_sample/sample_50/num_epochs_180/dynamic_normalisation/mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 180 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 10 --temperature_epoch_scaling True --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_50/num_epochs_180/dynamic_normalisation/mov_avg --weight_decay 1e-4

# Experiment 28: Sample an architecture after 100 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 100 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 dynamic_temperature True --exponential_moving_average True --gpuid 0 --init_weights True \
            --learning_rate 1e-4 --log_dir ./outputs/batch_sample/sample_100/num_epochs_180/dynamic_normalisation/mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 180 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 10 --temperature_epoch_scaling True --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_100/num_epochs_180/dynamic_normalisation/mov_avg --weight_decay 1e-4

# NOTE: 
#   1. Without exponential moving average of accuracies
#   2. Number of epochs = 220
#   3. Normalisation = True
#   4. Dynamic Normalisation = True

# Experiment 29: Sample an architecture after 50 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 50 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 dynamic_temperature True --gpuid 0 --init_weights True --learning_rate 1e-4 \
            --log_dir ./outputs/batch_sample/sample_50/num_epochs_220/dynamic_normalisation/no_mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 220 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 10 --temperature_epoch_scaling True --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_50/num_epochs_220/dynamic_normalisation/no_mov_avg --weight_decay 1e-4

# Experiment 30: Sample an architecture after 100 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 100 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 dynamic_temperature True --gpuid 0 --init_weights True --learning_rate 1e-4 \
            --log_dir ./outputs/batch_sample/sample_100/num_epochs_220/dynamic_normalisation/no_mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 220 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 10 --temperature_epoch_scaling True --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_100/num_epochs_220/dynamic_normalisation/no_mov_avg --weight_decay 1e-4

# NOTE: 
#   1. With exponential moving average of accuracies
#   2. Number of epochs = 220
#   3. Normalisation = True
#   4. Dynamic Normalisation = True

# Experiment 31: Sample an architecture after 50 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 50 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 dynamic_temperature True --exponential_moving_average True --gpuid 0 --init_weights True \
            --learning_rate 1e-4 --log_dir ./outputs/batch_sample/sample_50/num_epochs_220/dynamic_normalisation/mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 220 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 10 --temperature_epoch_scaling True --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_50/num_epochs_220/dynamic_normalisation/mov_avg --weight_decay 1e-4

# Experiment 32: Sample an architecture after 100 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 100 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 dynamic_temperature True --exponential_moving_average True --gpuid 0 --init_weights True \
            --learning_rate 1e-4 --log_dir ./outputs/batch_sample/sample_100/num_epochs_220/dynamic_normalisation/mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 220 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 10 --temperature_epoch_scaling True --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_100/num_epochs_220/dynamic_normalisation/mov_avg --weight_decay 1e-4

# NOTE: 
#   1. Without exponential moving average of accuracies
#   2. Number of epochs = 250
#   3. Normalisation = True
#   4. Dynamic Normalisation = True

# Experiment 33: Sample an architecture after 50 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 50 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 dynamic_temperature True --gpuid 0 --init_weights True --learning_rate 1e-4 \
            --log_dir ./outputs/batch_sample/sample_50/num_epochs_250/dynamic_normalisation/no_mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 250 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 10 --temperature_epoch_scaling True --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_50/num_epochs_250/dynamic_normalisation/no_mov_avg --weight_decay 1e-4

# Experiment 34: Sample an architecture after 100 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 100 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 dynamic_temperature True --gpuid 0 --init_weights True --learning_rate 1e-4 \
            --log_dir ./outputs/batch_sample/sample_100/num_epochs_250/dynamic_normalisation/no_mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 250 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 10 --temperature_epoch_scaling True --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_100/num_epochs_250/dynamic_normalisation/no_mov_avg --weight_decay 1e-4

# NOTE: 
#   1. With exponential moving average of accuracies
#   2. Number of epochs = 250
#   3. Normalisation = True
#   4. Dynamic Normalisation = True

# Experiment 35: Sample an architecture after 50 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 50 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 dynamic_temperature True --exponential_moving_average True --gpuid 0 --init_weights True \
            --learning_rate 1e-4 --log_dir ./outputs/batch_sample/sample_50/num_epochs_250/dynamic_normalisation/mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 250 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 10 --temperature_epoch_scaling True --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_50/num_epochs_250/dynamic_normalisation/mov_avg --weight_decay 1e-4

# Experiment 36: Sample an architecture after 100 batches
python ./main.py --architecture_search True --batch_norm True --batch_sampling_size 100 --batch_size 128 \
            --batch_update True --criterion_type cross-entropy --data_path ./datasets/cifar100/ --dir_name vgg19 \
            --discount_factor 0.9 --dropout 0.5 dynamic_temperature True --exponential_moving_average True --gpuid 0 --init_weights True \
            --learning_rate 1e-4 --log_dir ./outputs/batch_sample/sample_100/num_epochs_250/dynamic_normalisation/mov_avg --model_config E --model_name vgg19 \
            --normalise_prob_dist True --num_classes 100 --num_configs 100 --num_epochs 250 --num_repeats 3 \
            --num_val_examples 2000 --num_workers 2 --optimizer_type Adam --progress True --prob_dist maximum \
            --seed 0 --temperature 10 --temperature_epoch_scaling True --track_running_stats True \
            --visualisation_dir ./visualisation/batch_sample/sample_100/num_epochs_250/dynamic_normalisation/mov_avg --weight_decay 1e-4