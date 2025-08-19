import comet
import sacrebleu
from comet import download_model, load_from_checkpoint
import json
#from sacrebleu.metrics import BLEU
#from sacrebleu.tokenizers import MosesDetokenizer
import evaluate
import argparse
from fuzzywuzzy import fuzz
import stanza
import os
import json
import re
import time
import numpy as np


def argparse_args():
    parser = argparse.ArgumentParser(description="Evaluate model predictions.")
    parser.add_argument("--gold", type=str, help="Path to the gold jsonl.")
    parser.add_argument("--pred", type=str,  help="Path to the predictions file.")
    parser.add_argument("--output", type=str,  help="Path to save the evaluation results.")
    parser.add_argument("--tgt_lang", type=str, help="target language")
    parser.add_argument("--config", type=str, help="random_terms, proper_terms")
    parser.add_argument("--lemma", type=int, help="specify if lemmatization is needed for term success rate.")
    return parser.parse_args()

def preprocess_predictions(translations, gold):
    '''
    Creates a tuple of id src, ref, and pred from the translations.
    Sort them by id and return a dictionary of src, ref, and pred as keys.
    '''
    term_success=[]
    all_translations = []
    translations_dict = {'src': [], 'ref': [], 'pred': []}
    with open(translations, 'r', encoding='utf-8') as f, open(gold, 'r', encoding='utf-8') as g:
        for line in g:
            instance = json.loads(line)
            translations_dict['src'].append(instance['en'])
        current_set = False
        for line in f:
            line_ = line.strip()
            
            if line_.startswith('S-'):
                current_set = True
                S_id, src = line_.split('\t')
                tup = (int(S_id[2:]), )
            
            elif line_.startswith('T-'):
                T_id, tgt = line_.split('\t')
                tup += (tgt,)
            
            elif line_.startswith('H-'):
                H_id, lgprob, pred = line_.split('\t')
                tup += (pred,)
                all_translations.append(tup)
                tup=()
            else:
                continue

    all_translations.sort(key=lambda x: x[0])

    for tup in all_translations:
        #translations_dict['src'].append(tup[1])
        translations_dict['ref'].append(tup[1])
        translations_dict['pred'].append(tup[2])
    
    with open(gold, 'r', encoding='utf-8') as g:
    #print(translations_dict['src'])
        for predicted_sent, gold_inst in zip(translations_dict['pred'], g):
            gold_inst_=json.loads(gold_inst)
            gold_inst_['pred']=predicted_sent
            term_success.append(gold_inst_)
        
    return translations_dict, term_success

def format_data(translations_dict):
    '''
    Reformats the translations_dict into a list of dictionaries for comet scoring in the following format:
    [{'src': 'source text', 'mt': 'machine translation', 'ref': 'reference text'}, ...]
    '''
    src = translations_dict['src']
    #print(src)
    mt = translations_dict['pred']
    ref= translations_dict['ref']
    formatted_data = []
    for src, mt, ref in zip(src, mt, ref):
        formatted_data.append({'src': src, 'mt': mt, 'ref': ref})
    return formatted_data

def comet_score(translated_data, comet_da='Unbabel/wmt22-comet-da'):
    '''
    Computes the COMET-DA and COMET-QE scores for the translated data.
    '''
    formatted_data = format_data(translated_data)
    model = load_from_checkpoint(download_model(comet_da))
    model_output = model.predict(formatted_data, batch_size=32, gpus=1)
    score = model_output.system_score


    return score

def terminology_accuracy(gold_jsonl, src_lang, translations_dict):
    total_terms = 0
    predictions = translations_dict['pred']
    total_terms_appeared = 0
    gold_jsonl_list=[]
    with open(gold_jsonl, "r", encoding='utf-8') as f:
        for line in f:
            gold_jsonl_list.append(json.loads(line))

    for sample, prediction in zip(gold_jsonl_list, predictions):
        proper_terms = sample['proper_terms']
        prediction_set = set(prediction.split())
        for term in proper_terms:
            total_terms += 1
            if term in prediction_set or term.title() in prediction_set or term.lower() in prediction_set:
                total_terms_appeared += 1
    
    return total_terms_appeared / total_terms if total_terms > 0 else 0
    
def chrff(translations_dict):
    """
    Computes the chrF score for the translations.
    
    Args:
        translations_dict (dict): Dictionary containing 'src', 'ref', and 'pred' keys.
        
    Returns:
        float: The chrF score.
    """
    ref=  translations_dict['ref']
    references = [[r] for r in ref]
    predictions= translations_dict['pred']
    
    chrf = evaluate.load("chrf")
    chrf_score = chrf.compute(predictions=predictions, references=references, word_order=2, char_order=4, beta=2)
    return chrf_score
    


def stanza_lemmatize(sentence, tokenizer):
    doc = tokenizer(sentence)
    lemmas = [word.lemma for sent in doc.sentences for word in sent.words]
    lemmatized_sent = ' '.join(lemmas)
    return lemmatized_sent

def process_keywords_for_sentence(sentence, kw_list, lang, lemmatize=False, tokenizer=None):
    '''
    processing of a single sentence given its terminology.
    :param sentence: str, mt output
    :param kw_list: str (json), dictionary of keywords for a given sentence
    :param lang: str, language code
    :param lemmatize: bool, whether we apply the comparison of lemmatized words (important for morphologically-rich languages)
    :param tokenizer: stanza.Pipeline object, if lemmatize is True
    :returns kw_count: int, number of keywords
    :returns regex_count: int, number of regex matches
    :returns fuzzy90_count: int, number of fuzzy matches
    
    
    Question:
    lem_regex90, regex
    lem_fuzzy90, fuzzy90.
    Shouldnt there be 4 fields??

    '''
    #print(sentence)
    #print(kw_list)
    if kw_list is np.nan:
        return 0, 0, 0
    if sentence is np.nan:
        sentence = ''
    #kw_list = re.sub(',+', ',', kw_list)
    #kw_list = json.loads(kw_list)
    kw_count = len(kw_list)
    regex_count, fuzzy90_count, fuzzy70_count = 0, 0, 0
    if lemmatize:
        sentence = stanza_lemmatize(sentence, tokenizer)
    for kw_pair in kw_list:#{k1:v1, k2:v2}
        kw = kw_list[kw_pair]
        if lemmatize:
            kw = stanza_lemmatize(kw, tokenizer)
        match_regex =  find_match(sentence, kw, mode='regex')
        match_fuzzy90 = find_match(sentence, kw, mode='fuzzy', threshold=90)
        #match_fuzzy70 = find_match(sentence, kw, mode='fuzzy', threshold=50)

        regex_count += match_regex
        fuzzy90_count += match_fuzzy90

    return kw_count, regex_count, fuzzy90_count

def find_match(sentence, kw, mode='regex', threshold=90):
    '''
    matches a term within a sentence
    :param sentence: str
    :param kw: str, single term (may be MWE)
    :param mode: {'regex', 'fuzzy'} - whether we use regular expressions or FuzzyWuzzy to find a term
    :param threshold: [0-99], similarity threshold for FuzzyWuzzy to count it as a match (100 is complete match)
    :returns match: int, 1 if there is a match, 0 otherwise
    '''
    # regex match
    if mode == 'regex':
        #print(f'searching {kw} in {sentence}')
        try:
            regex_match = re.search(kw, sentence)
        #    print(regex_match)
        except:
            regex_match = re.search(re.escape(kw), re.escape(sentence))

        #print(regex_match)
        if regex_match is not None:
            match = 1
        else:
            match = 0

    # fuzzy match
    elif mode == 'fuzzy':
        fuzzy_match = fuzz.partial_ratio(kw, sentence)
        fuzzy_type = 'fuzzy'+str(threshold)
        if len(sentence) > 200:
            fuzzy_match = fuzzy_match * 2
        if fuzzy_match >= threshold:
            match = 1
        else:
            match = 0

    return match
    
def terminology_success_rate(term_success, config, lemmatize=False, tokenizer=None):
    '''
    The organizers' script's input is sentence, and keyword list per sentence. Hence, I should create a dictionary with src sentence, keyword_list, pred and tgt
    '''
    total_kw = 0
    
    regex_total=0
    fuzzy90_total=0
    lem_regex_total=0
    lem_fuzzy90_total=0
    
    for instance in term_success:
        sentence = instance['pred']
        kw_list = instance[config]
        kw_count, regex_count, fuzzy90_count = process_keywords_for_sentence(sentence, kw_list, "en", lemmatize, tokenizer)
        total_kw+=kw_count
        regex_total+=regex_count
        fuzzy90_total+=fuzzy90_count
        
    
    print(f'regex90: {regex_total/total_kw}')
    print(f'fuzzy90: {fuzzy90_total/total_kw}')
    




def output_bleu (output, gold, pred):
    """
    Compute mean BLEU score for a list of references and candidates.
    
    Args:
        gold (list): List of reference summaries
        pred (list): List of candidate summaries
        
    Returns:
        float: Mean BLEU score
    """
    all_gold_detokenized = []
    all_cand_detokenized = []
    with open(output, 'w', encoding='utf-8') as f:
        for ref, cand in zip(gold, pred):
            # Assuming each line is a separate prediction
            print(ref)
            print(cand)
            #detokenize the reference and candidate
            #ref = md.detokenize(ref.lower())
            #cand = md.detokenize(cand.lower())
            f.write(f'Reference: {ref}\n')
            f.write(f'Candidate: {cand}\n')
            #print(ref)
            #print(cand)
            bleu_score = sacrebleu.sentence_bleu(cand, [ref])
            f.write(f'BLEU score: {bleu_score.score}\n')
            f.write('\n')
        corpus_bleu = sacrebleu.corpus_bleu(pred, [gold])
        print(corpus_bleu.score)
        f.write(f'Corpus BLEU: {corpus_bleu.score}')


if __name__ == "__main__":
    args = argparse_args()
    data_dict, term_success = preprocess_predictions(args.pred, args.gold)
    print(data_dict)
    #formatted_data = format_data(data_dict) #output is a list of dictionaries
    comet_da = comet_score(data_dict, comet_da='Unbabel/wmt22-comet-da')
    print(f'comet_da: {comet_da}')
    chrff = chrff(data_dict)
    print(f'chrf: {chrff}')
    
   
    stanza_ = stanza.Pipeline(args.tgt_lang, processors='tokenize,lemma', lemma_pretagged=True, tokenize_pretokenized=False)
    #stanza_de = stanza.Pipeline('de', processors='tokenize,lemma', lemma_pretagged=True, tokenize_pretokenized=False)

    #stanza_ru = stanza.Pipeline('ru', processors='tokenize,lemma', lemma_pretagged=True, tokenize_pretokenized=False)
    #stanza_es = stanza.Pipeline('es', processors='tokenize,lemma', lemma_pretagged=True, tokenize_pretokenized=False)
    
    #lang_dict = {'de': stanza_de, 'ru': stanza_ru, 'es': stanza_es}
    if args.lemma == 1:
        print("lemmatize")
        terminology_success_rate(term_success, args.config, lemmatize=True, tokenizer=stanza_)
    else:
        terminology_success_rate(term_success, args.config, lemmatize=False, tokenizer=None)
        
    
    
        
    
    
        
    