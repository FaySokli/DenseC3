# DenseRetrievalMoE
To reproduce the experiments complete the following steps:

## Step 1:
Run the following command to complete data pre-processing and bring all datasets to the BEIR format: <br>
```python3 data_preprocessing.py```

## Step 2:
Update the parameters in the ```pipeline.sh``` file to the desired ones and execute the script. <br>
For example, the following command will train SB_MoE on TinyBERT using the HotpotQA dataset by employing 6 experts: <br>
```python3 1_train_new_moe.py model=tinybert dataset=hotpotqa testing=hotpotqa model.adapters.use_adapters=True model.adapters.num_experts_to_use=6 model.adapters.num_experts=6```