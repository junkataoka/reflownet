#!/usr/bin/zsh -l


# python preprocess.py

# python main.py --n_gpus=1 --num_nodes=1 --batch_size=8 --epochs=50 --neptune_logger --lr=0.01 --source_only \
# 			--src_input_file=source_input.pt --src_target_file=source_target.pt \
# 			--tar_input_file=target_input_cv1_train.pt --tar_target_file=target_target_cv1_train.pt \
# 			--model_path checkpoints/lstm_ac_reg_noda.ckpt --out_model_path checkpoints/lstm_ac_reg_noda.ckpt --neptune_logger

python main.py --n_gpus=1 --num_nodes=1 --batch_size=8 --epochs=50 --neptune_logger --lr=0.01 --source_only \
			--src_input_file=source_input.pt --src_target_file=source_target.pt \
			--tar_input_file=target_input_cv1_train.pt --tar_target_file=target_target_cv1_train.pt \
			--model_path checkpoints/lstm_ac_reg_noda_singlef.ckpt --out_model_path checkpoints/lstm_ac_reg_noda_singlef.ckpt \
			--neptune_logger --single_f

# python main.py --n_gpus=1 --num_nodes=1 --batch_size=2 --epochs=50 --neptune_logger --lr=0.001 \
# 			--src_input_file=source_input.pt --src_target_file=source_target.pt \
# 			--tar_input_file=target_input_cv1_train.pt --tar_target_file=target_target_cv1_train.pt \
# 			--model_path checkpoints/lstm_ac_reg_noda.ckpt --out_model_path checkpoints/lstm_ac_reg_mmd_cv1.ckpt --retrain --neptune_logger

# python main.py --n_gpus=1 --num_nodes=1 --batch_size=2 --epochs=50 --neptune_logger --lr=0.001 \
# 			--src_input_file=source_input.pt --src_target_file=source_target.pt \
# 			--tar_input_file=target_input_cv2_train.pt --tar_target_file=target_target_cv2_train.pt \
# 			--model_path checkpoints/lstm_ac_reg_noda.ckpt --out_model_path checkpoints/lstm_ac_reg_mmd_cv2.ckpt --retrain --neptune_logger

# python main.py --n_gpus=1 --num_nodes=1 --batch_size=2 --epochs=50 --neptune_logger --lr=0.001 \
# 			--src_input_file=source_input.pt --src_target_file=source_target.pt \
# 			--tar_input_file=target_input_cv3_train.pt --tar_target_file=target_target_cv3_train.pt \
# 			--model_path checkpoints/lstm_ac_reg_noda.ckpt --out_model_path checkpoints/lstm_ac_reg_mmd_cv3.ckpt --retrain --neptune_logger

# python main.py --n_gpus=1 --num_nodes=1 --batch_size=1 --epochs=50 --neptune_logger --lr=0.01 --source_only \
# 			--src_input_file=source_input.pt --src_target_file=source_target.pt \
# 			--tar_input_file=target_input_cv1_test.pt --tar_target_file=target_target_cv1_test.pt \
# 			--model_path checkpoints/lstm_ac_reg_noda.ckpt --out_model_path checkpoints/lstm_ac_reg_noda.ckpt --run_type test


# python main.py --n_gpus=1 --num_nodes=1 --batch_size=1 --epochs=50 --neptune_logger --lr=0.001 \
# 			--src_input_file=source_input.pt --src_target_file=source_target.pt \
# 			--tar_input_file=target_input_cv1_test.pt --tar_target_file=target_target_cv1_test.pt \
# 			--model_path checkpoints/lstm_ac_reg_mmd_cv1.ckpt --out_model_path checkpoints/lstm_ac_reg_mmd_cv1.ckpt --retrain --run_type test

# python main.py --n_gpus=1 --num_nodes=1 --batch_size=1 --epochs=50 --neptune_logger --lr=0.01 --source_only \
# 			--src_input_file=source_input.pt --src_target_file=source_target.pt \
# 			--tar_input_file=target_input_cv2_test.pt --tar_target_file=target_target_cv2_test.pt \
# 			--model_path checkpoints/lstm_ac_reg_noda.ckpt --out_model_path checkpoints/lstm_ac_reg_noda.ckpt --run_type test


# python main.py --n_gpus=1 --num_nodes=1 --batch_size=1 --epochs=50 --neptune_logger --lr=0.001 \
# 			--src_input_file=source_input.pt --src_target_file=source_target.pt \
# 			--tar_input_file=target_input_cv2_test.pt --tar_target_file=target_target_cv2_test.pt \
# 			--model_path checkpoints/lstm_ac_reg_mmd_cv2.ckpt --out_model_path checkpoints/lstm_ac_reg_mmd_cv2.ckpt --retrain --run_type test

# python main.py --n_gpus=1 --num_nodes=1 --batch_size=1 --epochs=50 --neptune_logger --lr=0.01 --source_only \
# 			--src_input_file=source_input.pt --src_target_file=source_target.pt \
# 			--tar_input_file=target_input_cv3_test.pt --tar_target_file=target_target_cv2_test.pt \
# 			--model_path checkpoints/lstm_ac_reg_noda.ckpt --out_model_path checkpoints/lstm_ac_reg_noda.ckpt --run_type test


# python main.py --n_gpus=1 --num_nodes=1 --batch_size=1 --epochs=50 --neptune_logger --lr=0.001 \
# 			--src_input_file=source_input.pt --src_target_file=source_target.pt \
# 			--tar_input_file=target_input_cv3_test.pt --tar_target_file=target_target_cv2_test.pt \
# 			--model_path checkpoints/lstm_ac_reg_mmd_cv3.ckpt --out_model_path checkpoints/lstm_ac_reg_mmd_cv2.ckpt --retrain --run_type test