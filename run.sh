#!/usr/bin/zsh -l


# srun python preprocess.py
python main.py --n_gpus=1 --num_nodes=1 --batch_size=3 --epochs=100 --neptune_logger --lr=0.01 --source_only \
			--src_input_file=source_input.pt --src_target_file=source_target.pt \
			--tar_input_file=target_input_cv1_train.pt --tar_target_file=target_target_cv1_train.pt \
			--model_path checkpoints/lstm_ac_noda.ckpt --out_model_path checkpoints/lstm_ac_reg_noda.ckpt


# srun python main.py --n_gpus=2 --num_nodes=1 --batch_size=1 --epochs=50 --neptune_logger --lr=0.001 \
# 			--src_input_file=source_input.pt --src_target_file=source_target.pt \
# 			--tar_input_file=target_input_cv1_train.pt --tar_target_file=target_target_cv1_train.pt \
# 			--model_path checkpoints/lstm_ac_reg_noda.ckpt --out_model_path checkpoints/lstm_ac_reg_mmd_cv1.ckpt --retrain


# srun python main.py --n_gpus=2 --num_nodes=1 --batch_size=1 --epochs=50 --neptune_logger --lr=0.001 \
# 			--src_input_file=source_input.pt --src_target_file=source_target.pt \
# 			--tar_input_file=target_input_cv2_train.pt --tar_target_file=target_target_cv2_train.pt \
# 			--model_path checkpoints/lstm_ac_reg_noda.ckpt --out_model_path checkpoints/lstm_ac_reg_mmd_cv2.ckpt --retrain


# srun python main.py --n_gpus=2 --num_nodes=1 --batch_size=1 --epochs=50 --neptune_logger --lr=0.001 \
# 			--src_input_file=source_input.pt --src_target_file=source_target.pt \
# 			--tar_input_file=target_input_cv3_train.pt --tar_target_file=target_target_cv3_train.pt \
# 			--model_path checkpoints/lstm_ac_reg_noda.ckpt --out_model_path checkpoints/lstm_ac_reg_mmd_cv3.ckpt --retrain