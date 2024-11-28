from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
import numpy as np
import torch
import os
import fitz #pdf reader
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm
import pandas as pd
import pickle
from statistics import mean

def remove_quotes(sentence):
    # Strip leading and trailing quotation marks
    if sentence.startswith('"') and sentence.endswith('"'):
        sentence = sentence[1:-1]
    return sentence

def removekey(d, key):
    r = dict(d)
    del r[key]
    return r

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def extract_text_from_pdf(pdf_path):
    text = ""

    try:
        with fitz.open(pdf_path) as pdf_document:
            num_pages = pdf_document.page_count

            for page_num in range(num_pages):
                page = pdf_document[page_num]
                text += page.get_text()

    except Exception as e:
        print(f"Error: {e}")

    return text

def semantic_morphological_check_1(subp, list_of_sentences, dataset_knowledge_1, dict_knowledge_1, top_results, limit_semantic_low, limit_semantic_high, num_retrieves):
    sentences_retrieved = []
    for i in range(0,num_retrieves):
        try:
            dict_knowledge_1[dataset_knowledge_1[list_of_sentences[top_results[1][i]]][0]]
        except:
            continue
        #check if the sentence that contains the proposition retrieved add some value
        if (top_results[0][i] > limit_semantic_high) and dataset_knowledge_1[list_of_sentences[top_results[1][i]]][2] == 1:
            continue
        #Check semantic and morphological similarity - 1
        elif (list_of_sentences[top_results[1][i]].lower() not in subp.lower()) and (top_results[0][i] > limit_semantic_low):
            sentences_retrieved.append(list_of_sentences[top_results[1][i]])

    return sentences_retrieved

def semantic_check_2(subp, input_sentence, sentence_retrieved, dataset_knowledge_1, dict_knowledge_1, model_encoder_only, model_decoder_only, limit_semantic_doc_retrieved):
    retrieved = False
    
    #check if the input sentence is too much different respect propositions of the retrieved document
    propositions_text_for_enrichment = dict_knowledge_1[dataset_knowledge_1[sentence_retrieved][0]]
    emb_propositions_text_for_enrichment = model_encoder_only.encode(propositions_text_for_enrichment)
    input_embedding = model_encoder_only.encode(input_sentence)
    cos_score_input_vs_retrieved_props = util.pytorch_cos_sim(input_embedding, emb_propositions_text_for_enrichment)[0]
    if min(cos_score_input_vs_retrieved_props) > limit_semantic_doc_retrieved:
        retrieved = True
        #check if the input sentence is too much different respect the retrieved document (ci sono casi in cui la frase intera è più diversa dalla frase input rispetto che le proposition)
        emb_text_for_enrichment = model_encoder_only.encode(dataset_knowledge_1[sentence_retrieved][0])
        cos_score_input_vs_retrieved_doc = util.pytorch_cos_sim(input_embedding, emb_text_for_enrichment)[0]
        if cos_score_input_vs_retrieved_doc > (limit_semantic_doc_retrieved*(1.2)):
            retrieved = True
        else:
            retrieved = False 
    else:
        retrieved = False
        
    return retrieved

def retrieve_documents_v2(input_sentence, list_subsentences, model_encoder_only, knowledge_embedding, dataset_knowledge_1, dict_knowledge_1, list_of_sentences, model_decoder_only, limit_semantic_low, limit_semantic_high, num_retrieves, limit_semantic_doc_retrieved):
    retrieved = False
    sentences_for_enrichment = []
    for subp in list_subsentences:
        count = 0
        subp_embedding = model_encoder_only.encode(subp)
        cos_scores = util.pytorch_cos_sim(subp_embedding, knowledge_embedding)[0]
        top_results = torch.topk(cos_scores, k=num_retrieves)
        sentences_retrieved = semantic_morphological_check_1(subp, list_of_sentences, dataset_knowledge_1, dict_knowledge_1, top_results, limit_semantic_low, limit_semantic_high, num_retrieves)
        number_of_retrieved = len(sentences_retrieved)
        if number_of_retrieved == 0:
            sentences_for_enrichment.append((subp, ''))
        else:
            for sentence_retrieved in sentences_retrieved:
                count += 1
                response = semantic_check_2(subp, input_sentence, sentence_retrieved, dataset_knowledge_1, dict_knowledge_1, model_encoder_only, model_decoder_only, limit_semantic_doc_retrieved)
                if response == False: 
                    if count == number_of_retrieved:
                        sentences_for_enrichment.append((subp, ''))
                    else:
                        continue
                    continue
                else:
                    sentences_for_enrichment.append((subp, sentence_retrieved))
                    retrieved = True     
                    break
        
    return retrieved, sentences_for_enrichment

def enrichment_candidates(list_input_and_sentences_for_enrichment_true, model_decoder_only, num_candidates):
    
    all_candidates = []
    prompts = ["As an expert in finance, you are tasked with enriching the original sentence provided with additional assertions, ensuring these specifics: "
            + "1. DO NOT CHANGE subjects of the original sentence"
            + "2. AVOID factual inaccuracies, logical errors and paradoxes"
            + "3. DO NOT CHANGE the SENTIMENT of the orginal sentence"
            + "4. DO NOT expand the scope of the original setence and DO NOT ADD unverifiable information"
            + "So, act as an expert in finance: could you enrich the original sentence '" + input_sentence
            + "' taking into consideration these assertions '" + sentences_for_enrichment + "'. "
            + "The assertions CANNOT be used to add the cause of the original sentence but to add POSSIBLE cause (e.g. cannot add 'due to ...' or 'because ...' but can add 'probably due to ...' or 'probably bacause ...'); they must clarify and provide as much detail as possible of the original sentence."
            + "Please report only and only the enriched sentence, as follows: "
            + "\"sentence enriched\"."
            + "Please report only and only the enriched sentence, enclosed within inverted commas."
            + "DO NOT PROVIDE FURTHER INFORMATION DESPITE THE ENRICHED SENTENCE ENCLOSED WITHIN INVERTED COMMAS." for input_sentence, sentences_for_enrichment in list_input_and_sentences_for_enrichment_true]
    
    mean_num_tokens = mean([len(i[0]) for i in list_input_and_sentences_for_enrichment_true])
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=(mean_num_tokens*len(list_input_and_sentences_for_enrichment_true)*(num_candidates*1.3)))
    outputs = model_decoder_only.generate(prompts, sampling_params)

    # Print the outputs
    for output in outputs:
        prompt = output.prompt
        response = output.outputs[0].text
        try:
            enriched_sentences = re.findall('"([^"]*)"', response)
            if len(enriched_sentences) < (num_candidates*0.2):
                response = response.replace('\n', '')
                enriched_sentences = re.split(r'(?<=[a-zA-Z\s])\.(?=[a-zA-Z0-9\s])', response)
                if len(enriched_sentences) > (num_candidates*1.5):
                    enriched_sentences_temp = [f'{enriched_sentences[i]} {enriched_sentences[i+1]}' for i in range(0, len(enriched_sentences)-1) if enriched_sentences[i][0].isdigit() == True]
                    enriched_sentences = enriched_sentences_temp
                    
        except:
            response = response.replace('\n', '')
            enriched_sentences = re.split(r'(?<=[a-zA-Z\s])\.(?=[a-zA-Z0-9\s])', response)
            if len(enriched_sentences) > (num_candidates*1.5):
                enriched_sentences_temp = [f'{enriched_sentences[i]} {enriched_sentences[i+1]}' for i in range(0, len(enriched_sentences)-1) if enriched_sentences[i][0].isdigit() == True]
                enriched_sentences = enriched_sentences_temp
            if len(enriched_sentences) < (num_candidates*0.2):
                print(f'error!!! check enriched_sentences: {enriched_sentences}')
                print(f'error!!! check response: {response}')
        all_candidates.append(enriched_sentences)
        
    return all_candidates, outputs

def main():

    #define path
    path = 'your_path'

    #ENCODER ONLY
    model_name_encoder_only = "intfloat/e5-base-v2"
    model_encoder_only = SentenceTransformer(model_name_encoder_only)

    #ENCODER ONLY SUBSENTENCE
    #import sentences already splitted
    with open(f'{path}/DATA/file_subsentences_densex_fiqa.pkl', 'rb') as f:
        dict_subsentences = pickle.load(f)

    #DECODER ONLY
    torch.cuda.empty_cache()
    model_name_decoder_only = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    model_decoder_only = LLM(model=model_name_decoder_only, quantization="gptq", dtype="half")

    # dataset for the knowledge: https://huggingface.co/datasets/infCapital/investopedia_terms_en

    set_of_sentences=set()

    #dataset_knowledge_1 = load_dataset("infCapital/investopedia_terms_en")
    with open(f'{path}/DATA_SHARED/file_subsentences_densex_investopedia.pkl', 'rb') as f:
        dataset_knowledge_1 = pickle.load(f)
        
    #add in knowledge also sentences from books of finance
    with open(f'{path}/DATA_SHARED/file_subsentences_densex_fin_books.pkl', 'rb') as f:
        dataset_knowledge_1_2 = pickle.load(f)
        
    dataset_knowledge_1.update(dataset_knowledge_1_2)

    #add also the new invstopedia dataset
    with open(f'{path}/DATA_SHARED/file_subsentences_densex_investopedia_2.pkl', 'rb') as f:
        dataset_knowledge_1_3 = pickle.load(f)

    dataset_knowledge_1.update(dataset_knowledge_1_3)

    #add also the new finrad dataset
    with open(f'{path}/DATA_SHARED/file_subsentences_densex_finrad.pkl', 'rb') as f:
        dataset_knowledge_1_4 = pickle.load(f)

    dataset_knowledge_1.update(dataset_knowledge_1_4)
    
    for proposition in tqdm(dataset_knowledge_1.keys()):
        set_of_sentences.add(proposition)


    #create dictionary of knowledge   
    with open(f'{path}/DATA_SHARED/file_doc_to_subsentences_densex_investopedia.pkl', 'rb') as f:
        dict_knowledge_1 = pickle.load(f)
        
    #add in dict of knowledge also sentences from books of finance
    with open(f'{path}/DATA_SHARED/file_doc_to_subsentences_densex_fin_books.pkl', 'rb') as f:
        dict_knowledge_1_2 = pickle.load(f)  

    dict_knowledge_1.update(dict_knowledge_1_2)

    #add also the new invstopedia dataset
    with open(f'{path}/DATA_SHARED/file_doc_to_subsentences_densex_investopedia_2.pkl', 'rb') as f:
        dict_knowledge_1_3 = pickle.load(f)

    dict_knowledge_1.update(dict_knowledge_1_3)

    #add also the finrad dataset
    with open(f'{path}/DATA_SHARED/file_doc_to_subsentences_densex_finrad.pkl', 'rb') as f:
        dict_knowledge_1_4 = pickle.load(f)

    dict_knowledge_1.update(dict_knowledge_1_4)

    #add in dict of knowledge also other document in the training and test dataset
    dict_knowledge_2 = {}
    for values in dict_subsentences.values():
        document_text = values[0]
        propostions_texts = values[1]
        dict_knowledge_2[document_text] = propostions_texts
           
    list_of_sentences = list(set_of_sentences)
    print(f'NUMBER OF PHRASES FOR THE KNOWLEDGE: {len(list_of_sentences)}')
    knowledge_embedding = model_encoder_only.encode(list_of_sentences)

    #ENRICH SENTENCES

    #evaluation on https://huggingface.co/datasets/ChanceFocus/fiqa-sentiment-classification
    dataset_origin = load_dataset("ChanceFocus/fiqa-sentiment-classification")
    dataset_sentences = []
    dataset_labels = []
    for i in dataset_origin['train']:
        if i['score']>0.3:
            dataset_sentences.append(i['sentence'])
            dataset_labels.append(2)
        elif i['score']<-0.3:
            dataset_sentences.append(i['sentence'])
            dataset_labels.append(0)
            
    for i in dataset_origin['valid']:
        if i['score']>0.3:
            dataset_sentences.append(i['sentence'])
            dataset_labels.append(2)
        elif i['score']<-0.3:
            dataset_sentences.append(i['sentence'])
            dataset_labels.append(0)
            
    for i in dataset_origin['test']:
        if i['score']>0.3:
            dataset_sentences.append(i['sentence'])
            dataset_labels.append(2)
        elif i['score']<-0.3:
            dataset_sentences.append(i['sentence'])
            dataset_labels.append(0)

    count_enrichment = 0
    count_no_retrieve = 0
    count_no_enrichment_to_diff = 0
    retrieved_bin = True
    X = []

    #save also sentences enriched and how they where enriched
    list_sentence_enriched = []
    list_elements_of_enrichment = []

    #SET PARAMS
    #tot k-propositions (proposition from the knowledge) retrieved for each i-proposition (proposition from the input)
    #alfa
    num_retrieves = 30
    #minimum semantic similarity between k-proposition and i-proposition
    #beta
    limit_semantic_low = 0.80
    #trheshold for checking if k-document (original document of the knowledge from which it was extracted the proposition) is equal to i-proposition; because, in this case, the retrievel is useless and maybe worst
    #gamma
    limit_semantic_high = 0.95

    #check if i-document (original document of the input from which it was extracted the proposition) is not too much different respect k-propositions and k-document (for this second check limit_semantic_doc_retrieved is multiplied by 1.2)
    #epsilon
    limit_semantic_doc_retrieved = 0.70

    #tot candidates enriched sentences generated by decoder-only
    #zeta
    num_candidates = 1
    #level of proximity of the enriched sentence to chose respect to the k-document retrieved choosen

    #check if documents from knowledge are already retrieved
    check_presence_knowledge_retrieved = False
    os.chdir(f'{path}')
    directory = os.fsencode('DATA')
        
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if 'file_input_and_sentences_for_enrichment_fiqa.pkl' == filename:
            check_presence_knowledge_retrieved = True
            break
        
    if check_presence_knowledge_retrieved == False:
        list_input_and_sentences_for_enrichment = []
        list_input_and_sentences_for_enrichment_true = []

        for index, input in tqdm(enumerate(dataset_sentences)):
            dataset_knowledge_temp = dataset_knowledge_1
            dict_knowledge_temp = dict_knowledge_1
            input_sentence = input
            for key, value in dict_knowledge_1.items():
                if input_sentence == key:
                    dict_knowledge_temp = removekey(dict_knowledge_1, key)
                    for value_1 in dict_knowledge_1[input_sentence]:
                        try:
                            dataset_knowledge_temp = removekey(dataset_knowledge_temp, value_1)
                        except:
                            continue
                    break 
            
            #list_subsentences = create_subsentences(model_subsentence, tokenizer_model_subsentence, input_sentence)
            list_subsentences = dict_knowledge_2[input_sentence]
            retrieved_bin, sentences_for_enrichment = retrieve_documents_v2(input_sentence, list_subsentences, model_encoder_only, knowledge_embedding, dataset_knowledge_temp, dict_knowledge_temp, list_of_sentences, model_decoder_only, limit_semantic_low, limit_semantic_high, num_retrieves, limit_semantic_doc_retrieved)
            if retrieved_bin == False:
                count_no_retrieve += 1
                list_input_and_sentences_for_enrichment.append([input_sentence, retrieved_bin, ''])
            else:
                sentence_for_enrichment = ''
                for proposition in sentences_for_enrichment:
                    try:
                        text_for_enrichment = dataset_knowledge_1[proposition[1]][0]
                        if text_for_enrichment in sentence_for_enrichment:
                            continue
                        else:
                            sentence_for_enrichment = sentence_for_enrichment + '\n' + text_for_enrichment
                    except:
                        continue
                    
                list_input_and_sentences_for_enrichment.append([input_sentence, retrieved_bin, sentence_for_enrichment])
                list_input_and_sentences_for_enrichment_true.append((input_sentence, sentence_for_enrichment))
                #save document retrieved
                # create a binary pickle file 
                f = open(f"{path}/DATA/file_input_and_sentences_for_enrichment_fiqa.pkl","wb")
                # write the python object (dict) to pickle file
                pickle.dump(list_input_and_sentences_for_enrichment,f)
                # close file
                f.close()

                # create a binary pickle file 
                f = open(f"{path}/DATA/file_input_and_sentences_for_enrichment_true_fiqa.pkl","wb")
                # write the python object (dict) to pickle file
                pickle.dump(list_input_and_sentences_for_enrichment_true,f)
                # close file
                f.close()
    else:
        list_input_and_sentences_for_enrichment_true = pickle.load(open(f'{path}/DATA/file_input_and_sentences_for_enrichment_true_fiqa.pkl', 'rb'))
        print(f'knowledge retrieved computed yet')
        
    #check if possible candidates have been already created
    check_presence_candidates = False
    os.chdir(f'{path}')
    directory = os.fsencode('DATA')
        
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if f'file_{num_candidates}_candidates_for_enrichment_fiqa.pkl' == filename:
            check_presence_candidates = True
            break
        
    if check_presence_candidates == False:
        all_candidates, outputs = enrichment_candidates(list_input_and_sentences_for_enrichment_true, model_decoder_only, num_candidates)
        list_sentence_to_fix = []
        index_to_modify = []
        for index, i in enumerate(all_candidates):
            if len(i) > (num_candidates*1.5) or len(i) < (num_candidates*0.3):
                list_sentence_to_fix.append(list_input_and_sentences_for_enrichment_true[index])
                index_to_modify.append(index)
        print(f"try to do again {len(list_sentence_to_fix)} wrong output")
        all_candidates_fixed, outputs = enrichment_candidates(list_sentence_to_fix, model_decoder_only, num_candidates)  
        #update candidates 
        for index, i in enumerate(index_to_modify):
            if len(all_candidates_fixed[index]) < 1:
                continue
            else:
                all_candidates[i] = all_candidates_fixed[index]
            
        list_input_and_sentences_candidates = []
        for i in range(0, len(list_input_and_sentences_for_enrichment_true)):
            input_sentence = list_input_and_sentences_for_enrichment_true[i][0]
            sentence_for_enrichment = list_input_and_sentences_for_enrichment_true[i][1]
            candidates = all_candidates[i]
            list_input_and_sentences_candidates.append((input_sentence, sentence_for_enrichment, candidates))

        #save possible enriched sentences
        # create a binary pickle file 
        f = open(f"{path}/DATA/file_{num_candidates}_candidates_for_enrichment_fiqa.pkl","wb")
        # write the python object (dict) to pickle file
        pickle.dump(list_input_and_sentences_candidates,f)
        # close file
        f.close()
    else:
        list_input_and_sentences_candidates = pickle.load(open(f"{path}/DATA/file_{num_candidates}_candidates_for_enrichment_fiqa.pkl", 'rb'))
        print(f'candidates computed yet')
        
    list_final_enriched = []
    for input_sentence, sentence_for_enrichment, candidates  in tqdm(list_input_and_sentences_candidates):
        if len(candidates) == 0:
            list_final_enriched.append((input_sentence, sentence_for_enrichment, candidates, False, input_sentence)) 
        else:
            list_final_enriched.append((input_sentence, sentence_for_enrichment, candidates, True, candidates[0]))
        
    list_sentence_original = []
    for input_sentence, sentence_for_enrichment, candidates, enriched, sentence_enriched in list_final_enriched:
        if enriched == False:
            count_no_enrichment_to_diff += 1
        else:   
            count_enrichment += 1
            list_sentence_original.append(input_sentence)
            sentence_enriched = remove_quotes(sentence_enriched)
            list_sentence_enriched.append(sentence_enriched)
            list_elements_of_enrichment.append(sentence_for_enrichment)
            
    #SAVE NEW DATASET
    data_sentences_enriched = {'original_sentence': list_sentence_original,
            'sentence': list_sentence_enriched, 
            }

    # Create DataFrame
    output_sentences_enriched = pd.DataFrame(data_sentences_enriched)
    output_sentences_enriched = output_sentences_enriched.drop_duplicates()
    output_sentences_enriched = output_sentences_enriched.drop_duplicates(subset=['original_sentence'])

    fiqa_enriched = {'original_sentence': dataset_sentences,
            'label': dataset_labels
            }
    output_fiqa_enriched = pd.DataFrame(fiqa_enriched)
    output_fiqa_enriched_temp = output_fiqa_enriched.merge(output_sentences_enriched, on = 'original_sentence', how = 'left')
    output_fiqa_enriched_temp.sentence.fillna(output_fiqa_enriched_temp.original_sentence, inplace=True)
    output_fiqa_enriched = output_fiqa_enriched_temp[['sentence', 'label']]
    output_fiqa_enriched.to_csv(f'{path}/OUTPUT/fiqa_enriched_allagree_ablation_no_post_retrieval.csv')

if __name__ == '__main__':
    main()