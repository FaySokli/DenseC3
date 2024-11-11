import json
import pandas as pd

file_path = '/home/ubuntu/esokli/DenseRetrievalMoE/multi-domain/computer_science/collection.jsonl'
corpus = pd.read_json(file_path, lines=True)
corpus.rename(columns={'id': '_id'}, inplace=True)
corpus['_id'] = corpus['_id'].astype(str)
corpus.to_json(file_path, orient='records', lines=True)

query_ids = queries_train['_id']
rel_doc_ids = queries_train['rel_doc_ids']

rows = []
for query_id, doc_ids in zip(query_ids, rel_doc_ids):
    for doc_id in doc_ids:
        rows.append({'query-id': query_id, 'corpus-id': doc_id, 'score': 1})

qrels_df = pd.DataFrame(rows)

qrels_df.to_csv('/home/ubuntu/esokli/DenseRetrievalMoE/multi-domain/computer_science/train/qrels.tsv', sep='\t', index=False)   
