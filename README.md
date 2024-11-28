# RE-FIN
## RE-FIN: Retrieval-based Enrichment for Financial data

RE-FIN (Retrieval-based Enrichment for FINancial data) is an automated system designed to retrieve information from a knowledge base to enrich financial sentences, making them more knowledge-dense and explicit. RE-FIN generates propositions from the knowledge base and employs Retrieval-Augmented Generation (RAG) to augment the original text with relevant information. 
A large language model (LLM) rewrites the original sentence, incorporating this data. Since the LLM does not create new content, the risk of hallucinations is significantly reduced. The LLM generates multiple new sentences using relevant information from the knowledge base.

## RE-FIN FRAMEWORK

![Diagram RE-FIN](https://github.com/user-attachments/assets/77dd4bb8-97d3-4dd1-9dbd-2600cdc22d0b)

A traditional RAG process includes three main phases indexing, retrieval, and generation; moreover, an advanced RAG method, like the one proposed in the paper, also employs pre-retrieval and post-retrieval strategies. 

- **Indexing** starts with the cleaning and extraction of raw data in PDF, CSV, and TSV formats and converts them into a uniform plain text format. To accommodate the context limitations of language models, text is segmented into sentences delimited by points, becoming smaller and digestible chunks.

- **Pre-retrieval process**. In this stage, the primary focus is optimizing the indexing structure and the original query. Optimizing indexing aims to enhance the quality of the content being indexed. We involve a very little-used strategy proposed by Chen et al., 2023 (Dense X Retrieval: What Retrieval Granularity Should We Use?), enhancing data granularity, and optimizing index structures. We choose propositions as a retrieval unit since the retrieved texts are more condensed with information relevant to the original sentence, reducing the need for lengthy input tokens and minimizing the inclusion of extraneous, irrelevant information. Propositions are then encoded into vector representations using an embedding model and stored in a vector database. This step enables efficient similarity searches in the subsequent retrieval phase.

- **Retrieval**. Upon receipt of a user query, the RAG system employs the same encoding model utilized during the indexing phase to transform the query into a vector representation. It then computes the similarity scores between the query vector and the vector of chunks within the indexed corpus. The system prioritizes and retrieves the top K chunks that demonstrate the greatest similarity to the query. These chunks are subsequently used as the expanded context in the prompt.

- **Post-Retrieval Process**. Once the relevant context is retrieved, it’s crucial to integrate it effectively with the query. The main methods in the post-retrieval process include re-ranking chunks and context compressing. In particular, we utilize an innovative heuristic process to create a new sentence similar to the original one that includes the most relevant documents retrieved.

- **Generation**. In this phase, the best sentence enriched created is corrected using an LLM and used as the final version of the sentence.

## **Deployment**

Use Python 3.10.12

Example with Financial PhraseBank data

1. Run FPB/PREPROCESS/compute_subsentences_of_fpb_densex.py to compute propositions for the input dataset
2. Run FPB/main.py to perform the enrichment

After creating the enriched dataset, you can reproduce the ablation study by running the scripts located in the ABLATION folder. 
Additionally, you can replicate the evaluation experiments described in the original published article.

## References
```
@article{malandri2025re-fin,
  title={RE-FIN: Retrieval-based Enrichment for Financial data},
  author={Malandri, Lorenzo and Mercorio, Fabio and Pallucchini, Filippo},
  journal={The 31st International Conference on Computational Linguistics (COLING)},
  year={2025},
  publisher={Association for Computational Linguistics 2025}
}
```


