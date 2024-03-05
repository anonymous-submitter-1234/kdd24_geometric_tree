mkdir pretrain_scripts
cd pretrain_scripts
dataset_name="5x_glia";
model_name="GTMP";
mkdir ${model_name}
cd ${model_name}
python3 pretrain_distance.py --model ${model_name} \
						--data_dir ./data/sneuro_exp_cond \
           				--save_dir ./results \
						--dataset Neuron_exp_cond \
						--dataset_name ${dataset_name} \
						--num_epochs 100 \
						--batch_size 16
cd ..
cd ..
