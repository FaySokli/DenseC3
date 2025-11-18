# Leveraging Cognitive Complexity of Texts for Contextualization in Dense Retrieval

To reproduce the experiments conducted for the evaluation of DenseC3:

## Step 1:
Run the following command to complete data pre-processing and bring all datasets to the BEIR format: <br>
<em>python3 data_preprocessing.py</em>

For our experiments, we use the following seven publicly available IR benchmarks: 
- MSMARCO, HotpotQA, and Natural Questions from the BEIR collection;
- TREC Deep Learning Track 2019 & 2020;
- Political Science and Computer Science collections from the Multi-Domain Benchmark [1]. We bring these last two collections to the BEIR format for our experiments.

[1] Bassani, E., Kasela, P., Raganato, A., & Pasi, G. (2022, October). A multi-domain benchmark for personalized search evaluation. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management (pp. 3822-3827).

## Step 2:
Following Li et al. [2], we train a multi-head BERT model into a multi-label classifier of text into the six Cognitive Complexity levels of Bloom's Taxonomy using the code and data as provided by the authors (https://github.com/SteveLEEEEE/EDM2022CLO/blob/main/Multi-Label.ipynb).
We add two methods to the original code: "predict_labels" for our complexity distribution plots and "predict_logits" for producing all document logits. <br>

We provide the code to produce the document logits for all five collections (once the classifier is trained): <br>
<em>python3 multi_label.py</em>

[2] Yuheng Li, Mladen Rakovic, Boon Xin Poh, Dragan Gasevic, and Guanliang Chen. 2022. Automatic classification of learning objectives based on bloom’s taxonomy. In Proceedings of the 15th International Conference on Educational Data Mining, EDM 2022, Durham, UK, July 24-27, 2022. International Educational Data Mining Society.

## Step 3:
Update the parameters in the pipeline.sh file to the desired ones and execute the script.

For instance, the following command will train the <em>DenseC3_w</em> variant on TinyBERT using the HotpotQA dataset: <br>
<em>python3 1_train_new_moe.py model=tinybert dataset=hotpotqa testing=hotpotqa model.adapters.use_adapters=True model.init.specialized_mode=densec3_w</em>

*Note: Only two specialized modes are allowed; (i) 'densec3_top1' and (ii) 'densec3_w'.
