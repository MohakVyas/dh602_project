#!/bin/bash

# Define the hyperparameter search space
declare -a lr=("0.001")
declare -a new_lr=("5e-6" "1e-6")
declare -a dec_hsz=("128" "256" "512")
declare -a dropout=("0.3" "0.5" "0.7")
declare -a rnn_layers=("1")
declare -a batch_size=("16" "32" "64")

# Set the fixed value for epochs
epochs="50"
count=0
# Iterate over hyperparameters
for lr_val in "${lr[@]}"; do
    for new_lr_val in "${new_lr[@]}"; do
        for dec_hsz_val in "${dec_hsz[@]}"; do
            for dropout_val in "${dropout[@]}"; do
                for rnn_layers_val in "${rnn_layers[@]}"; do
                    for batch_size_val in "${batch_size[@]}"; do
                        # Construct the command to execute
                        ((count++))
                        # if [[ "$count" == '1' ]]; then
                        #     continue
                        # fi 

                        # if [[ "$count" == '2' ]]; then
                        #     continue
                        # fi 

                        # if [[ "$count" == '3' ]]; then
                        #     continue
                        # fi 

                        cd results_swin
                        results_path="resultsA2C_${lr_val}_${new_lr_val}_${dec_hsz_val}_${dropout_val}_${rnn_layers_val}_${batch_size_val}"
                        logs_path="logs_${lr_val}_${new_lr_val}_${dec_hsz_val}_${dropout_val}_${rnn_layers_val}_${batch_size_val}"
                        models_path="models_${lr_val}_${new_lr_val}_${dec_hsz_val}_${dropout_val}_${rnn_layers_val}_${batch_size_val}"

                        cmd1=`mkdir $results_path`
                        # echo "Executing: $cmd1"
                        eval $cmd1
                        cd $results_path

                        cmd2= `echo --lr $lr_val --new_lr $new_lr_val --dec_hsz $dec_hsz_val --dropout $dropout_val --rnn_layers $rnn_layers_val --batch_size $batch_size_val --epochs $epochs --training_results logs/$logs_path/training_results.txt --pretrain_actor_logs logs/$logs_path/pretrain_actor_logs.txt --pretrain_critic_logs logs/$logs_path/pretrain_critic_logs.txt --results_path results/$results_path --actor_pretrained models/$models_path/actor_pretrained.pth --critic_pretrained models/$models_path/critic_pretrained.pth --actor_path models/$models_path/actor.pth --critic_path models/$models_path/critic.pth --enc_dec_path models/$models_path/enc_dec.pth --final_results_path final_results/$final_results_path --train_logs logs/$logs_path/actor_critic_train_logs.txt --train_reward_logs logs/$logs_path/actor_critic_train_reward_logs.txt > README.txt`
                        # echo "Executing: $cmd2"
                        eval $cmd2
                        cd ..

                        cd ../logs_swin
                        
                        cmd4=`mkdir $logs_path`
                        # echo "Executing: $cmd4"
            
                        eval $cmd4
                        cd $logs_path

                        # echo "Executing: $cmd2"
                        cmd2= `echo --lr $lr_val --new_lr $new_lr_val --dec_hsz $dec_hsz_val --dropout $dropout_val --rnn_layers $rnn_layers_val --batch_size $batch_size_val --epochs $epochs --training_results logs/$logs_path/training_results.txt --pretrain_actor_logs logs/$logs_path/pretrain_actor_logs.txt --pretrain_critic_logs logs/$logs_path/pretrain_critic_logs.txt --results_path results/$results_path --actor_pretrained models/$models_path/actor_pretrained.pth --critic_pretrained models/$models_path/critic_pretrained.pth --actor_path models/$models_path/actor.pth --critic_path models/$models_path/critic.pth --enc_dec_path models/$models_path/enc_dec.pth --final_results_path final_results/$final_results_path --train_logs logs/$logs_path/actor_critic_train_logs.txt --train_reward_logs logs/$logs_path/actor_critic_train_reward_logs.txt > README.txt`
                        eval $cmd2
                        cd ..

                        cd ../models_swin


                        
                        cmd6=`mkdir $models_path`
                        # echo "Executing: $cmd6"
                        eval $cmd6
                        cd $models_path


                        # echo "Executing: $cmd2"
                        cmd2= `echo --lr $lr_val --new_lr $new_lr_val --dec_hsz $dec_hsz_val --dropout $dropout_val --rnn_layers $rnn_layers_val --batch_size $batch_size_val --epochs $epochs --training_results logs/$logs_path/training_results.txt --pretrain_actor_logs logs/$logs_path/pretrain_actor_logs.txt --pretrain_critic_logs logs/$logs_path/pretrain_critic_logs.txt --results_path results/$results_path --actor_pretrained models/$models_path/actor_pretrained.pth --critic_pretrained models/$models_path/critic_pretrained.pth --actor_path models/$models_path/actor.pth --critic_path models/$models_path/critic.pth --enc_dec_path models/$models_path/enc_dec.pth --final_results_path final_results/$final_results_path --train_logs logs/$logs_path/actor_critic_train_logs.txt --train_reward_logs logs/$logs_path/actor_critic_train_reward_logs.txt > README.txt`
                        eval $cmd2
                        cd ..
                        
                        cd ../final_results_swin
                        final_results_path="final_results_${lr_val}_${new_lr_val}_${dec_hsz_val}_${dropout_val}_${rnn_layers_val}_${batch_size_val}"
                        mkdir $final_results_path
                        cd $final_results_path
                        cmd2= `echo --lr $lr_val --new_lr $new_lr_val --dec_hsz $dec_hsz_val --dropout $dropout_val --rnn_layers $rnn_layers_val --batch_size $batch_size_val --epochs $epochs --training_results logs/$logs_path/training_results.txt --pretrain_actor_logs logs/$logs_path/pretrain_actor_logs.txt --pretrain_critic_logs logs/$logs_path/pretrain_critic_logs.txt --results_path results/$results_path --actor_pretrained models/$models_path/actor_pretrained.pth --critic_pretrained models/$models_path/critic_pretrained.pth --actor_path models/$models_path/actor.pth --critic_path models/$models_path/critic.pth --enc_dec_path models/$models_path/enc_dec.pth --final_results_path final_results/$final_results_path --train_logs logs/$logs_path/actor_critic_train_logs.txt --train_reward_logs logs/$logs_path/actor_critic_train_reward_logs.txt > README.txt`
                        eval $cmd2
                        cd ..
                        cd ..

                        cmd="python train.py --lr $lr_val --new_lr $new_lr_val --dec_hsz $dec_hsz_val --dropout $dropout_val --rnn_layers $rnn_layers_val --batch_size $batch_size_val --epochs $epochs --training_results logs_swin/$logs_path/training_results.txt --pretrain_actor_logs logs_swin/$logs_path/pretrain_actor_logs.txt --pretrain_critic_logs logs_swin/$logs_path/pretrain_critic_logs.txt --results_path results_swin/$results_path --actor_pretrained models_swin/$models_path/actor_pretrained.pth --critic_pretrained models_swin/$models_path/critic_pretrained.pth --actor_path models_swin/$models_path/actor.pth --critic_path models_swin/$models_path/critic.pth --enc_dec_path models_swin/$models_path/enc_dec.pth --final_results_path final_results_swin/$final_results_path --train_logs logs_swin/$logs_path/actor_critic_train_logs.txt --train_reward_logs logs_swin/$logs_path/actor_critic_train_reward_logs.txt --pretrain_enc_dec_logs logs_swin/$logs_path/pretrain_enc_dec_logs.txt" 
                        
                        # Print and execute the command
                        echo "Executing: $cmd"
                        eval $cmd
                        
                    done
                done
            done
        done
    done
done
