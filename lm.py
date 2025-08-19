import nltk
from nltk.util import ngrams
from nltk.lm import MLE, Laplace, KneserNeyInterpolated
#https://www.nltk.org/api/nltk.lm.html
from nltk.lm.preprocessing import padded_everygram_pipeline
import argparse
import pickle
import json
nltk.download('punkt')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm_train", type=str, help="Path to the outdomain training corpus.")
    parser.add_argument("--model_type", type=str, choices=["MLE", "Laplace", "KneserNey"], help="Type of language model to train.")
    parser.add_argument("--n", type=int, help="Order of the n-gram model.")
    parser.add_argument("--outmodel", type=str, help="Path to save the trained out-domain model.")
    parser.add_argument("--indomaindatade", type=str, help="path to german in-domain data for training.")
    parser.add_argument("--indomaindataes", type=str, help="Path to the spanish in-domain data for training.")
    parser.add_argument("--indomaindataru", type=str, help="Path to the russian in-domain data for training.")
    parser.add_argument("--inmodel", type=str, help="Path to save the trained in-domain model.")
    return parser.parse_args()

def get_indomain_data(indomainde, indomaines, indomainru):
    in_domain_source_data = []
    with open(indomainde, 'r', encoding='utf-8') as f, open(indomaines, 'r', encoding='utf-8') as f2, open(indomainru, 'r', encoding='utf-8') as f3:
        indomain_data_de = json.load(f)
        indomain_data_es = json.load(f2)
        indomain_data_ru = json.load(f3)
    for data,data2,data3  in zip(indomain_data_de, indomain_data_es, indomain_data_ru):
        for integer1, integer2, integer3 in zip(data, data2, data3):
          
            source_sentence_de  = data[integer1]['en']
            source_sentence_es  = data2[integer2]['en']
            source_sentence_ru  = data3[integer3]['en']
       
            in_domain_source_data.append(nltk.word_tokenize(source_sentence_de.strip().lower()))
            in_domain_source_data.append(nltk.word_tokenize(source_sentence_es.strip().lower()))
            in_domain_source_data.append(nltk.word_tokenize(source_sentence_ru.strip().lower()))
    print(len(in_domain_source_data), "in-domain sentences loaded.")
    return in_domain_source_data

def tokenize_text(text):
    """
    Tokenizes the input text into sentences and words.
    """
    tokenized_sentences = []
    #loop through text
    with open(text, 'r') as f:
        for line in f:
            source_sentence = line.strip().split('\t')[0]
            
            # Tokenize the source sentence
            tokens = nltk.word_tokenize(source_sentence.lower())
            #print(tokens)
            tokenized_sentences.append(tokens)

    #print(len(tokenized_sentences), "sentences tokenized.")
    print(len(tokenized_sentences))
    return tokenized_sentences

if __name__ == "__main__":
    args = get_args()
    # Tokenize the text
    tokenized_text = tokenize_text(args.lm_train)
    tokenized_in_domain_data = get_indomain_data(args.indomaindatade, args.indomaindataes, args.indomaindataru) 


    # Prepare the data for n-gram model
    train_data, padded_sents = padded_everygram_pipeline(args.n, tokenized_text)
    in_domain_data, padded_in_domain_sents = padded_everygram_pipeline(args.n, tokenized_in_domain_data)


    # Choose the model type
    if args.model_type == "MLE":
        out_model = MLE(args.n)
        in_model = MLE(args.n)
    elif args.model_type == "Laplace":
        out_model = Laplace(args.n)
        in_model = Laplace(args.n)
    elif args.model_type == "KneserNey":
        out_model = KneserNeyInterpolated(args.n)
        in_model = KneserNeyInterpolated(args.n)

    # Fit the model
    out_model.fit(train_data, padded_sents)
    in_model.fit(in_domain_data, padded_in_domain_sents)
    print(out_model.vocab)
    
    

    #save the n-gram model
    with open(args.outmodel, 'wb') as f:
        pickle.dump(out_model, f)
    with open(args.inmodel, 'wb') as f:
        pickle.dump(in_model, f)
    
    #model.cross

 