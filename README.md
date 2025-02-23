# Leveraging Cognitive Complexity of Texts for Contextualization in Dense Retrieval

To reproduce the experiments complete the following steps:

## Step 1:
Run the following command to complete data pre-processing and bring all datasets to the BEIR format: <br>
<em>python3 data_preprocessing.py</em>

For our experiments we use the following four publically available IR benchmarks: (i) HotpotQA and (ii) Natural Questions from the BEIR collection, as well as the (iii) Political Science and (iv) Computer Science collections from the Multi-Domain Benchmark. We bring the last two collections to the BEIR format for our experiments.

## Step 2:
Following Li et al. [1], we train a multi-head BERT model into a multi-label classifier of text into the six Cognitive Complexity levels of Bloom's Taxonomy using the code and data as provided by the authors (https://github.com/SteveLEEEEE/EDM2022CLO/blob/main/Multi-Label.ipynb).
We add two methods in the original code, namely "predict_labels" and "predict_logits", for our plots and document logits, respectively. <br><br>

We provide the code to produce the document logits for the all four collections (once the classifier is trained): <br>
<em>python3 multi_label.py</em>

[1] Yuheng Li, Mladen Rakovic, Boon Xin Poh, Dragan Gasevic, and Guanliang Chen. 2022. Automatic classification of learning objectives based on bloom’s taxonomy. In Proceedings of the 15th International Conference on Educational Data Mining, EDM 2022, Durham, UK, July 24-27, 2022. International Educational Data Mining Society.

## Step 3:
Update the parameters in the pipeline.sh file to the desired ones and execute the script.

For instance, the following command will train the <em>Blooms_all</em> variant on TinyBERT using the HotpotQA dataset: <br>
<em>python3 1_train_new_moe.py model=tinybert dataset=hotpotqa testing=hotpotqa model.adapters.use_adapters=True model.init.specialized_mode=blooms_all</em>

*Note: Only two specialized modes are allowed; (i) 'blooms_top1' and (ii) 'blooms_all'.