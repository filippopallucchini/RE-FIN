from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd
import torch
import json

def create_subsentences(model_subsentence, tokenizer_model_subsentence, input_sentence):
    title = ""
    section = ""

    input_text = f"Title: {title}. Section: {section}. Content: {input_sentence}"
    input_ids = tokenizer_model_subsentence(input_text, return_tensors="pt").input_ids
    outputs = model_subsentence.generate(input_ids.to(device), max_new_tokens=512).cpu()

    output_text = tokenizer_model_subsentence.decode(outputs[0], skip_special_tokens=True)

    try:
        output = json.loads(output_text)
    except:
        output = [input_sentence]
    
    return output

def main():
    #Define path
    path = 'your_path'

    #ENCODER ONLY SUBSENTENCE
    model_name_subsentence = "chentong00/propositionizer-wiki-flan-t5-large"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_model_subsentence = AutoTokenizer.from_pretrained(model_name_subsentence)
    model_subsentence = AutoModelForSeq2SeqLM.from_pretrained(model_name_subsentence).to(device)

    #ENRICH SENTENCES

    #evaluation on https://github.com/pyRis/SEntFiN
    path_sentfin = f'{path}/DATA/SEntFiN_clean.csv'
    #dataset_origin = pd.read_csv(path_sentfin)
    dataset = pd.read_csv(path_sentfin)

    list_elements_of_enrichment = []
    dict_subsentences = {}

    for index, input in tqdm(enumerate(dataset['sentence'])):
        input_sentence = input
        list_subsentences = create_subsentences(model_subsentence, tokenizer_model_subsentence, input_sentence)
        dict_subsentences[index] = (input_sentence , list_subsentences)
        list_elements_of_enrichment.append(list_subsentences)

    #save info of enrichment         
    data_sentences_enriched = {'input_sentence': dataset['sentence'],
            'element_for_enrichment': list_elements_of_enrichment
            }

    # Create DataFrame
    output_sentences_enriched = pd.DataFrame(data_sentences_enriched)
    output_sentences_enriched.to_csv(f'{path}/DATA/check_subsentences_densex_sentfin.csv')

    # create a binary pickle file 
    f = open(f"{path}/DATA/file_subsentences_densex_sentfin.pkl","wb")

    # write the python object (dict) to pickle file
    pickle.dump(dict_subsentences,f)

    # close file
    f.close()

if __name__ == '__main__':
    main()