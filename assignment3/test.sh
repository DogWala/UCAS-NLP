# --memory-efficient-fp16
# fairseq-generate D:\\大作业3\\nmt\\data-bin \
#     --path D:\\大作业3\\nmt\\model\\checkpoints\\checkpoint_best.pt \
#     --batch-size 128 --beam 8 > best_res2.txt \
#     --remove-bpe --cpu
file=original_tesnformer
fairseq-generate /home/wala/mt/nmt/data-bin/en-zh \
    --path /home/wala/mt/nmt/model/checkpoints7/checkpoint_best.pt \
    --remove-bpe --fp16 --scoring bleu \
    --batch-size 64 --beam 8 > $file.txt

# fairseq-interactive /home/wala/mt/nmt/data-bin/en-zh \
#     --input /home/wala/mt/nmt/data-bin/test.en \
#     --path /home/wala/mt/nmt/model/checkpoints4/checkpoint_best.pt \
#     --batch-size 1 --beam 8 --remove-bpe --scoring bleu > best_res.txt
