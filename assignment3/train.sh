# CUDA_VISIBLE_DEVICES=0 fairseq-train D:\\大作业3\\nmt\\data-bin --arch transformer \
# 	--source-lang en --target-lang zh \
#     --max-epoch 30 --batch-size 64 \
#     --optimizer adam  --lr 0.001 --adam-betas '(0.9, 0.98)' \
#     --lr-scheduler inverse_sqrt --max-tokens 4096  --dropout 0.3 \
#     --criterion label_smoothed_cross_entropy  --label-smoothing 0.1 \
#     --max-update 200000  --warmup-updates 4000 --warmup-init-lr '1e-07' \
#     --num-workers 4 \
#     --keep-best-checkpoints 8 \
# 	--save-dir model/checkpoints2 &

# this is the properite way to train
CUDA_VISIBLE_DEVICES=0 fairseq-train /home/wala/mt/nmt/data-bin/en-zh --arch transformer \
	--source-lang en --target-lang zh \
    --max-epoch 12 --batch-size 256 \
    --optimizer adam  --lr 0.0005 --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --max-tokens 1024  --dropout 0.3 \
    --criterion label_smoothed_cross_entropy  --label-smoothing 0.1 \
    --max-update 200000  --warmup-updates 4000 --warmup-init-lr '1e-07' \
    --num-workers 8 --tensorboard-logdir log7/ \
    --keep-best-checkpoints 3 \
	--save-dir model/checkpoints7 &

# CUDA_VISIBLE_DEVICES=0 fairseq-train /home/wala/mt/nmt/data-bin/en-zh --arch lstm \
# 	--source-lang en --target-lang zh \
#     --max-epoch 10 --batch-size 64 \
#     --optimizer adam  --lr 0.0005 --adam-betas '(0.9, 0.98)' \
#     --lr-scheduler inverse_sqrt --max-tokens 1024  --dropout 0.3 \
#     --criterion cross_entropy \
#     --max-update 200000  --warmup-updates 4000 --warmup-init-lr '1e-07' \
#     --num-workers 1 --tensorboard-logdir log5/ \
#     --keep-best-checkpoints 3 \
# 	--save-dir model/checkpoints6 &