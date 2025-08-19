import argparse
import string

'''
this script computes the probabilities per word.
Corpora is the indomain and outdomain data after CED scoring (lm.de.test.txt on google drive). The reason why we do this is so that we don't artificially remove important terms.
\Suppose that in our indomain data, we have a lot of technical high_freq words
'''

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--corpora", type=str, help="full corpora for CED selection")
    parser.add_argument("--outfreq", type=str, help="output of frequencies")

    return parser.parse_args()

def frequency(corpora):
    en_dictionary = {}
    total_words = 0
    with open(corpora, 'r') as id:
        for line in id:
            #print(line)
            try:
                en_sent, de_sent = line.strip().split('\t')
                en_sent = line.strip()
                #print(de_sent)
                en_sent = en_sent.strip()
                en_words = en_sent.split()
                for word in en_words:
                    word= word.lower().strip(string.punctuation)
                    #print(word)

                    total_words +=1
                    if word not in en_dictionary:
                        en_dictionary[word]=1
                    else:
                        en_dictionary[word]+=1
                #print(en_dictionary)
            except Exception as e:
                print(f'error processing {e}')

      
    
    #convert to probs
    for word in en_dictionary:
        frequency = en_dictionary[word]
        en_dictionary[word] = frequency/total_words
    

    return en_dictionary


if __name__ == "__main__":
    args = get_args()
    en_dictionary = frequency(args.corpora)
    with open(args.outfreq, 'w') as of:
        for word in en_dictionary:
            of.write(word + '\t' + str(en_dictionary[word]) + '\n')
            

    
    

                