file=original_transformer
grep ^T $file.txt | cut -f2- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $file.ref
grep ^H $file.txt | cut -f3- | perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' > $file.sys
fairseq-score -s $file.sys -r $file.ref
