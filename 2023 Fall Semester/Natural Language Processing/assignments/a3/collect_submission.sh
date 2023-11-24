rm -f assignment3_submission.zip
zip -r assignment3_submission.zip src/ birth_dev.tsv birth_places_train.tsv wiki.txt vanilla.model.params vanilla.finetune.params perceiver.finetune.params vanilla.nopretrain.dev.predictions vanilla.nopretrain.test.predictions vanilla.pretrain.dev.predictions vanilla.pretrain.test.predictions perceiver.pretrain.dev.predictions perceiver.pretrain.test.predictions
