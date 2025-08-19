import nltk
from nltk import word_tokenize, ngrams
import argparse
import pickle
import fasttext
nltk.download('punkt_tab')
model = fasttext.load_model('lid.176.bin')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, help="source language code")
    parser.add_argument("--tgt", type=str, help="target language code")
    parser.add_argument("--outmodel", type=str, help="Path to the outdomain n-gram model.")
    parser.add_argument("--inmodel", type =str, help="Path to the indomain n-gram model.")
    parser.add_argument("--testdata", type=str, help="Path to the test data for CED scoring.")
    parser.add_argument("--new_data", type=str, help="Path to the new data after CED scoring. Output of this script")
    return parser.parse_args()

def compute_CE(outmodel, inmodel, testdata, src, tgt):
    '''
    Returns a list of tuples containing the CED score, source/english sentence and the target sentence.
    '''
    scored_data = [] 
    with open(testdata, 'r', encoding = 'utf-8') as f:
        for line in f:
            try:
                source_sentence, target_sentence = line.strip().split('\t')
                #check if source_sentences / target sentences are valid or not
                prediction_source = model.predict(source_sentence, k=1)  # k=1 returns top prediction
                language_source = prediction_source[0][0].replace('__label__', '')
                confidence_source = prediction_source[1][0]
                
                prediction_target = model.predict(target_sentence, k=1)
                language_target = prediction_target[0][0].replace('__label__', '')
                confidence_target = prediction_target[1][0]
                
                if language_source!= src or language_target!= tgt:
                    continue
                
                source_tokens = word_tokenize(source_sentence.lower())
                padded_source_tokens = list(ngrams(source_tokens, n=5, pad_left=True, pad_right=True, 
                                                left_pad_symbol="<s>", right_pad_symbol="</s>"))
                # Compute CED score
                out_score = outmodel.entropy(padded_source_tokens)
                in_score = inmodel.entropy(padded_source_tokens)
                print(out_score, in_score)
                # Calculate CED score
                ced_score = in_score - out_score
                # Append the result
                scored_data.append((ced_score, source_sentence, target_sentence))

            except Exception as e:
                print(f"Error processing line: {e}")
                continue

    return scored_data

def save_scored_data(scored_data, output_path):
    '''
    Saves the sorted scored data to a file.
    '''

    sorted_scored_data = sorted(scored_data, key=lambda x: x[0])
    with open(output_path, 'w', encoding='utf-8') as f:
        for score, source, target in sorted_scored_data:
            f.write(f"{score}\t{source}\t{target}\n")
if __name__ == "__main__":
    args = get_args()
    # Load the in-domain model
    with open(args.inmodel, 'rb') as f:
        inmodel = pickle.load(f)

    # Load the out-domain model
    with open(args.outmodel, 'rb') as f:
        outmodel = pickle.load(f)
    # Compute CED scores
    scored_data = compute_CE(outmodel, inmodel, args.testdata, args.src, args.tgt)
    
    # Save the scored data
    save_scored_data(scored_data, args.new_data)
    


