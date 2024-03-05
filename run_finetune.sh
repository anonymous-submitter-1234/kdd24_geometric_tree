mkdir finetune_scripts
cd finetune_scripts
model_name=GTMP;
dataset_name="5x_glia";
pretrain_dataset="5x_glia";
mkdir ${model_name}
cd ${model_name}
mkdir ${dataset_name}
cd ${dataset_name}
python3 finetune.py --model ${model_name} \
						   --data_dir ./data/neuro_exp_cond \
						   --save_dir ./results \
						   --num_epochs 100 \
						   --batch_size 16 \
						   --dataset Neuron_exp_cond \
						   --model_name ${dataset_name} \
						   --dataset_name ${dataset_name}
cd ..
cd ..
cd ..
