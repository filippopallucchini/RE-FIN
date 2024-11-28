import numpy as np
from sklearn.metrics import accuracy_score
import torch
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm
import pandas as pd

def sentiment_classifier_standard_method_few_shot_learning(enriched_sentence, model_decoder_only):
    count = 0
    prompts = ["Act as a sentiment classifier expert in financial domain. Here we report 6 examples of classification of sentiment:\n"
           + "1) \'CRR cut to enable bullish equity, rupee market\' classified as \'POSITIVE\'.\n"
           + "2) \'Euro wobbles before Greek vote, Greek shares tumble\' classified as \'NEGATIVE\'.\n"
           + "3) \'Rupee ends day at 59.27 against US dollar\' classified as \'NEUTRAL\'.\n"
           + "4) \'United Spirits appoints Anand Kripalu as CEO\' classified as \'NEUTRAL\'.\n"
           + "5) \'Stock Buzz: Sun Pharma may face resistance at current levels\' classified as \'NEGATIVE\'.\n"
           + "6) \'Sugar companies & investors get the sweet smell of revival\' classified as \'POSITIVE\'.\n"  
           + "Determine the sentiment category of the financial news as \'NEGATIVE\', \'NEUTRAL\' or \'POSITIVE\': \'" + enriched_sentence + "\'"
           + "DO NOT PROVIDE FURTHER INFORMATION DESPITE THE SENTIMENT CATEGORY ENCLOSED WITHIN INVERTED COMMAS."
           ]
            
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=200)

    outputs = model_decoder_only.generate(prompts, sampling_params)

    # Print the outputs
    for output in outputs:
        prompt = output.prompt
        response = output.outputs[0].text
        #print(response)
    
    final_response = 3
    extract_sentiment = re.findall('"([^"]*)"', response)
    if len(extract_sentiment) > 0:
        if "neutral" in extract_sentiment[0].lower():
            final_response = 1
        elif "positive" in extract_sentiment[0].lower():
            final_response = 2
        elif "negative" in extract_sentiment[0].lower():
            final_response = 0
    else:
        if "NEUTRAL" in response:
            final_response = 1
        elif "POSITIVE" in response:
            final_response = 2
        elif "NEGATIVE" in response:
            final_response = 0
        elif "neutral" in response.lower():
            final_response = 1
        elif "positive" in response.lower():
            final_response = 2
        elif "negative" in response.lower():
            final_response = 0
        else:
            print(f'not found, so: NEUTRAL\n{response}\nThe sentence was:\n{enriched_sentence}')
            final_response = 1
            count = 1
        
    return final_response, count

def sentiment_classifier_standard_method(enriched_sentence, model_decoder_only):
    count = 0
    prompts = ["Act as a sentiment classifier expert in financial domain."
           + "Determine the sentiment category of the financial news as 'NEGATIVE', 'NEUTRAL' or 'POSITIVE': '" + enriched_sentence + "'"
           + "DO NOT PROVIDE FURTHER INFORMATION DESPITE THE SENTIMENT CATEGORY ENCLOSED WITHIN INVERTED COMMAS."
           ]
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=200)

    outputs = model_decoder_only.generate(prompts, sampling_params)

    # Print the outputs
    for output in outputs:
        prompt = output.prompt
        response = output.outputs[0].text
        #print(response)
        
    final_response = 3
    extract_sentiment = re.findall('"([^"]*)"', response)
    if len(extract_sentiment) > 0:
        if "neutral" in extract_sentiment[0].lower():
            final_response = 1
        elif "positive" in extract_sentiment[0].lower():
            final_response = 2
        elif "negative" in extract_sentiment[0].lower():
            final_response = 0
    else:
        if "NEUTRAL" in response:
            final_response = 1
        elif "POSITIVE" in response:
            final_response = 2
        elif "NEGATIVE" in response:
            final_response = 0
        elif "neutral" in response.lower():
            final_response = 1
        elif "positive" in response.lower():
            final_response = 2
        elif "negative" in response.lower():
            final_response = 0
        else:
            print(f'not found, so: NEUTRAL\n{response}\nThe sentence was:\n{enriched_sentence}')
            final_response = 1
            count = 1
        
    return final_response, count

def main():
    #define path
    path = 'your_path'

    #DECODER ONLY
    torch.cuda.empty_cache()
    model_name_decoder_only = "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"
    model_decoder_only = LLM(model=model_name_decoder_only, quantization="gptq", dtype="half")

    #PREDICT SENTIMENT
    dataset_path = f'{path}/OUTPUT/sentfin_enriched.csv'
    df = pd.read_csv(dataset_path)
    df = df[['sentence', 'label']]

    #compute prediction of sentences without enrichment

    #evaluation on https://github.com/pyRis/SEntFiN
    path_sentfin = f'{path}/DATA/SEntFiN_clean.csv'
    dataset = pd.read_csv(path_sentfin)

    dataset_temp = pd.DataFrame({'original': list(dataset['sentence'])})

    df_diff = df.merge(dataset_temp, how = 'left', left_index=True, right_index=True)

    df_diff['diff'] = df_diff.apply(lambda x: 1 if x['sentence'] != x['original'] else 0, axis = 1)

    df_diff = df_diff[df_diff['diff']==1]

    X = df_diff['sentence']
    y = df_diff['label']

    new_preds = []
    not_predicted_enr = 0
    for x in X:
        sentiment, count = sentiment_classifier_standard_method_few_shot_learning(x, model_decoder_only)
        #sentiment, count = sentiment_classifier_standard_method(x, model_decoder_only)
        new_preds.append(sentiment)
        not_predicted_enr += count

    accuracy_enr = accuracy_score(y, new_preds)
    print(f'Accuracy-Score enr: {accuracy_enr}')

    #X_old = dataset['train']['sentence']
    #y_old = dataset['train']['label']

    X_old = df_diff['original']
    y_old = df_diff['label']

    new_preds_no_enr = []
    not_predicted_no_enr = 0
    for x in X_old:
        sentiment, count = sentiment_classifier_standard_method_few_shot_learning(x, model_decoder_only)
        #sentiment, count = sentiment_classifier_standard_method(x, model_decoder_only)
        new_preds_no_enr.append(sentiment)
        not_predicted_no_enr += count

    accuracy_no_enr = accuracy_score(y_old, new_preds_no_enr)
    print(f'Accuracy-Score no enr: {accuracy_no_enr}')

    data = {'Accuracy enr': [accuracy_enr],
            'count not predicted enr': [not_predicted_enr],
            'Accuracy no enr': [accuracy_no_enr],
            'count not predicted no enr': [not_predicted_no_enr]
            }
    # Create DataFrame
    output = pd.DataFrame(data)
    output.to_csv(f'{path}/OUTPUT/output_financial_db_decoder_only_delta_sentfin.csv')

    #save sentences enriched

    data_sentences = {'old_sentence': X_old,
            'new_sentence': X, 
            'true_label': y,
            'predicted_label_no_enr': new_preds_no_enr,
            'predicted_label': new_preds
            }

    # Create DataFrame
    output_sentences = pd.DataFrame(data_sentences)
    output_sentences['delta_diff_vs_no_enr'] = output_sentences.apply(lambda x: 1 if x['predicted_label_no_enr']==x['predicted_label'] else 0, axis=1)
    output_sentences['delta_true_vs_enr'] = output_sentences.apply(lambda x: 1 if x['true_label']==x['predicted_label'] else 0, axis=1)
        
    output_sentences.to_csv(f'{path}/OUPTUT/check_sentences_enriched_for_decoder_only_delta_evaluator_sentfin.csv')

if __name__ == '__main__':
    main()