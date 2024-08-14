DATASET=nfcorpus
MODEL=contriever

python3 1_train_new_moe.py model=$MODEL dataset=$DATASET testing=$DATASET dataset.model_dir='output/'$DATASET'/saved_model_biencoder' training.lr=1e-5 training.max_epoch=30 training.batch_size=128
TESTING_DATASET=$DATASET
python3 2_create_embedding_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET dataset.model_dir='output/'$DATASET'/saved_model_biencoder' training.batch_size=64
echo "$DATASET MoE"
python3 3_test_biencoder_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET dataset.model_dir='output/'$DATASET'/saved_model_biencoder' training.batch_size=64


DATASET=fever
MODEL=contriever

python3 1_train_new_moe.py model=$MODEL dataset=$DATASET testing=$DATASET dataset.model_dir='output/'$DATASET'/saved_model_biencoder' training.lr=1e-4 training.max_epoch=10 training.batch_size=64
TESTING_DATASET=$DATASET
python3 2_create_embedding_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET dataset.model_dir='output/'$DATASET'/saved_model_biencoder' training.batch_size=64
echo "$DATASET MoE"
python3 3_test_biencoder_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET dataset.model_dir='output/'$DATASET'/saved_model_biencoder' training.batch_size=64

python3 1_train_new_moe.py model=$MODEL dataset=$DATASET testing=$DATASET dataset.model_dir='output/'$DATASET'/saved_model_biencoder' training.lr=1e-6 training.max_epoch=10 training.batch_size=64 model.adapters.use_adapters=False
python3 2_create_embedding_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET dataset.model_dir='output/'$DATASET'/saved_model_biencoder' training.batch_size=64 model.adapters.use_adapters=False
echo "$DATASET Zero"
python3 3_test_biencoder_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET dataset.model_dir='output/'$DATASET'/saved_model_biencoder' training.batch_size=64 model.adapters.use_adapters=False


DATASET=hotpotqa
MODEL=contriever

python3 1_train_new_moe.py model=$MODEL dataset=$DATASET testing=$DATASET dataset.model_dir='output/'$DATASET'/saved_model_biencoder' training.lr=1e-4 training.max_epoch=10 training.batch_size=64
TESTING_DATASET=$DATASET
python3 2_create_embedding_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET dataset.model_dir='output/'$DATASET'/saved_model_biencoder' training.batch_size=64
echo "$DATASET MoE"
python3 3_test_biencoder_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET dataset.model_dir='output/'$DATASET'/saved_model_biencoder' training.batch_size=64

python3 1_train_new_moe.py model=$MODEL dataset=$DATASET testing=$DATASET dataset.model_dir='output/'$DATASET'/saved_model_biencoder' training.lr=1e-6 training.max_epoch=10 training.batch_size=64 model.adapters.use_adapters=False
python3 2_create_embedding_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET dataset.model_dir='output/'$DATASET'/saved_model_biencoder' training.batch_size=64 model.adapters.use_adapters=False
echo "$DATASET Zero"
python3 3_test_biencoder_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET dataset.model_dir='output/'$DATASET'/saved_model_biencoder' training.batch_size=64 model.adapters.use_adapters=False


DATASET=nq-train
MODEL=contriever

python3 1_train_new_moe.py model=$MODEL dataset=$DATASET testing=$DATASET dataset.model_dir='output/'$DATASET'/saved_model_biencoder' training.lr=1e-4 training.max_epoch=10 training.batch_size=64
TESTING_DATASET=$DATASET
python3 2_create_embedding_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET dataset.model_dir='output/'$DATASET'/saved_model_biencoder' training.batch_size=64
echo "$DATASET MoE"
python3 3_test_biencoder_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET dataset.model_dir='output/'$DATASET'/saved_model_biencoder' training.batch_size=64

python3 1_train_new_moe.py model=$MODEL dataset=$DATASET testing=$DATASET dataset.model_dir='output/'$DATASET'/saved_model_biencoder' training.lr=1e-6 training.max_epoch=10 training.batch_size=64 model.adapters.use_adapters=False
python3 2_create_embedding_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET dataset.model_dir='output/'$DATASET'/saved_model_biencoder' training.batch_size=64 model.adapters.use_adapters=False
echo "$DATASET Zero"
python3 3_test_biencoder_moe.py model=$MODEL dataset=$TESTING_DATASET testing=$TESTING_DATASET dataset.model_dir='output/'$DATASET'/saved_model_biencoder' training.batch_size=64 model.adapters.use_adapters=False

