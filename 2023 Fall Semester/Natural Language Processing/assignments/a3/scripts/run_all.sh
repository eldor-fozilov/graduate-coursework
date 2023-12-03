# Note: Don't forget to edit the hyper-parameters for part d.

# Train on the names dataset
python src/run.py finetune vanilla wiki.txt \
--writing_params_path vanilla.model.params \
--finetune_corpus_path birth_places_train.tsv
# Evaluate on the dev set, writing out predictions
python src/run.py evaluate vanilla wiki.txt \
--reading_params_path vanilla.model.params \
--eval_corpus_path birth_dev.tsv \
--outputs_path vanilla.nopretrain.dev.predictions
# Evaluate on the test set, writing out predictions
python src/run.py evaluate vanilla wiki.txt \
--reading_params_path vanilla.model.params \
--eval_corpus_path birth_test_inputs.tsv \
--outputs_path vanilla.nopretrain.test.predictions


# Pretrain the model
python src/run.py pretrain vanilla wiki.txt \
        --writing_params_path vanilla.pretrain.params
        
# Finetune the model
python src/run.py finetune vanilla wiki.txt \
        --reading_params_path vanilla.pretrain.params \
        --writing_params_path vanilla.finetune.params \
        --finetune_corpus_path birth_places_train.tsv
        
# Evaluate on the dev set; write to disk
python src/run.py evaluate vanilla wiki.txt  \
        --reading_params_path vanilla.finetune.params \
        --eval_corpus_path birth_dev.tsv \
        --outputs_path vanilla.pretrain.dev.predictions
        
# Evaluate on the test set; write to disk
python src/run.py evaluate vanilla wiki.txt  \
        --reading_params_path vanilla.finetune.params \
        --eval_corpus_path birth_test_inputs.tsv \
        --outputs_path vanilla.pretrain.test.predictions

# Pretrain the model
python src/run.py pretrain perceiver wiki.txt --bottleneck_dim 64 \
        --pretrain_lr 6e-3 --writing_params_path perceiver.pretrain.params
        
# Finetune the model
python src/run.py finetune perceiver wiki.txt --bottleneck_dim 64 \
        --reading_params_path perceiver.pretrain.params \
        --writing_params_path perceiver.finetune.params \
        --finetune_corpus_path birth_places_train.tsv
        
# Evaluate on the dev set; write to disk
python src/run.py evaluate perceiver wiki.txt --bottleneck_dim 64 \
        --reading_params_path perceiver.finetune.params \
        --eval_corpus_path birth_dev.tsv \
        --outputs_path perceiver.pretrain.dev.predictions
        
# Evaluate on the test set; write to disk
python src/run.py evaluate perceiver wiki.txt --bottleneck_dim 64 \
        --reading_params_path perceiver.finetune.params \
        --eval_corpus_path birth_test_inputs.tsv \
        --outputs_path perceiver.pretrain.test.predictions