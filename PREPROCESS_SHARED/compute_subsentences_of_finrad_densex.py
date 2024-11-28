from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd
import re
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

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

def main():
    #Define path
    path = 'your_path'

    #ENCODER ONLY SUBSENTENCE
    model_name_subsentence = "chentong00/propositionizer-wiki-flan-t5-large"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer_model_subsentence = AutoTokenizer.from_pretrained(model_name_subsentence)
    model_subsentence = AutoModelForSeq2SeqLM.from_pretrained(model_name_subsentence).to(device)

    dataset = pd.read_csv(f'{path}/DATA_SHARED/finrad_clean.csv')

    list_name_of_document = []
    list_sentences = []
    list_elements_of_enrichment = []
    dict_subsentences = {}

    count_duplicates = 0

    for index, original_document in tqdm(enumerate(dataset['sentence'])):
        for index_2, input_sentence in enumerate(re.split(r'(?<=\.)[ \n]', original_document.replace('\n\n\n', ' ').replace('\n ', ' ').replace('\n', ' ').replace('\ufeff', ' ').replace('\xa0', ' ').replace('- ', '').replace('-', '').replace('e.g.', 'e.g').replace('et al. ', 'et al ').strip().lstrip())):
            if len(input_sentence) < 200 and has_numbers(input_sentence) == False:
                list_subsentences = create_subsentences(model_subsentence, tokenizer_model_subsentence, input_sentence)
                list_name_of_document.append('')
                list_sentences.append(input_sentence)
                list_elements_of_enrichment.append(list_subsentences)
                for subsentence in list_subsentences:
                    if subsentence in dict_subsentences.keys():
                        subsentence_new = 'Speaking about '+ '' + ', ' + subsentence
                        count_duplicates += 1
                        dict_subsentences[subsentence_new] = (input_sentence , '', len(list_subsentences))   
                    else:
                        dict_subsentences[subsentence] = (input_sentence , '', len(list_subsentences))
            else:
                continue
    try:
        dict_subsentences = removekey(dict_subsentences, '')
    except:
        print('no need to remove key ""')

    #save info of enrichment         
    data_sentences_enriched = {'name_of_document': list_name_of_document,
            'input_sentence': list_sentences,
            'element_for_enrichment': list_elements_of_enrichment
            }

    # Create DataFrame
    output_sentences_enriched = pd.DataFrame(data_sentences_enriched)
    output_sentences_enriched.to_csv(f'{path}/DATA_SHARED/check_subsentences_densex_finrad.csv')

    # create a binary pickle file 
    f = open(f"{path}/DATA_SHARED/file_subsentences_densex_finrad.pkl","wb")

    # write the python object (dict) to pickle file
    pickle.dump(dict_subsentences,f)

    # close file
    f.close()

    dict_doc_to_subp = {}
    for index in range(len(list_sentences)):
        dict_doc_to_subp[list_sentences[index]] = list_elements_of_enrichment[index]
        
    # create a binary pickle file 
    f = open(f"{path}/DATA_SHARED/file_doc_to_subsentences_densex_finrad.pkl","wb")

    # write the python object (dict) to pickle file
    pickle.dump(dict_doc_to_subp,f)

    # close file
    f.close()

if __name__ == '__main__':
    main()