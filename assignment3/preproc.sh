TEXT = /home/wala/mt/nmt/proced
fairseq-preprocess --source-lang en --target-lang zh \
    --worker 4 \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/en-zh