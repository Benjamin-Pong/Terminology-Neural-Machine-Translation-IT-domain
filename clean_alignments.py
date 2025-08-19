'''
Goal is to repurpose the alignment indices to account for multi-word terms. Terminologies are not just bijections
Input:  would be 2 files in parajoah form, and word form
output: 2 files in parajoah form, and word form

one-to-one: 
Surjection (en->):

removed duplicates, removed stopwords, accounted for multiword terms en->german direction

'''

import argparse
import nltk
import string
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Download tokenizer data if needed
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
print(stop_words)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ip",
        help="the path to the input parajoah form",
        required = True
    )
    parser.add_argument(
        "--iw",
        help="The path to the input word form",
        required=True
    )
    parser.add_argument(
        "--op",
        help="the path to the output parajoah form",
        required=True
    )
    parser.add_argument(
        "--ow",
        help="The path to the output word form",
        required=True
    )

    return parser.parse_args()


def preprocessing(ip, iw):
    ip_list = []
    iw_list = []
    with open(ip, 'r') as ip, open(iw, 'r') as iw:
        for ip_ in ip:
            ip_split = ip_.strip().split()
            #to remove duplicates
            ip_list.append(ip_split)
        for iw_ in iw:
            iw_split = iw_.strip().split()
            #to remove duplicates
            iw_list.append(iw_split)
    return ip_list, iw_list


def is_not_int_or_punct(s):
    # Remove whitespace
    s = s.strip()
    # Check if s is not empty, not all digits, and not all punctuation
    return bool(s) and not s.isdigit() and not all(c in string.punctuation for c in s)

def align(ip_list, iw_list):
    '''
    this method extracts purely one-to-one source-term matchings.

    delimiters '-' <-> '<SEP>'

    new delimiters for multiword terms: ~ <-> <W>
    '''
    all_new_word_alignments = []
    all_new_parajoah_alignments = []
    iteration = 0
    for ip_, iw_ in zip(ip_list,iw_list): #each line
        word_ende_dict = {}
        word_deen_dict = {}
        new_word_alignment_line = []
        new_parajoah_alignment_line = []
        print(ip_, iw_)
        '''
        this for-loop repurposes the alignments in terms of a dictionary of tuples
        (en_word, en_idx) : [(de_word, de_idx)...]
        '''
        surjective_word = None
        surjective_idx = None
        previous_de_word, previous_de_idx = None, None
        previous_en_word, previous_en_idx = None, None
        last_idx = len(ip_)-1
        current_idx = 0
        for a,b in zip(ip_,iw_):
            #split by delimiters
            
            print(a, b)
            en_word, de_word = b.split('<sep>')
            en_word = en_word.rstrip(string.punctuation)
            de_word = de_word.rstrip(string.punctuation)
            en_idx, de_idx = a.split('-')
            #previous_de_word, previous_de_idx = None, None
            print(iteration, en_word, de_word)
            
            '''
            if is_not_int_or_punct(en_word) and is_not_int_or_punct(de_word):
                if (en_word, en_idx) not in word_ende_dict:
                    word_ende_dict[(en_word, en_idx)] = [(de_word, de_idx)]
                else:
                    word_ende_dict[(en_word, en_idx)].append((de_word, de_idx))
                if (de_word, de_idx) not in word_deen_dict:
                    word_deen_dict[(de_word, de_idx)] = [(en_word, en_idx)]
                else:
                    word_deen_dict[(de_word, de_idx)].append((en_word, en_idx))
            '''

            if surjective_word is None and surjective_idx is None:
                surjective_word, surjective_idx = f'{en_word}', f'{en_idx}'
                print("surjective", surjective_word, surjective_idx)
            if previous_de_word is None:
                previous_de_word = de_word
                previous_de_idx = de_idx
                #surjective_word , surjective_idx = f'{de_word}', f'{de_idx}'
                #print(previous_de_idx, previous_de_word)
     
            elif previous_de_word:
                #check if current_de_word == previous_de_word:
                    #if yes, construct surjective word in en-de direction
                    #else, finish off surjective_word with de_word and de_idx, then replace new previous word
                if de_word == previous_de_word:
                    print('yes')
                    surjective_word = f'{surjective_word}<w>{en_word}'
                    surjective_idx = f'{surjective_idx}~{en_idx}'
                    current_idx+=1
                    print(surjective_word, surjective_idx)
                    if current_idx == last_idx:
                        surjective_word = f'{en_word}<sep>{de_word}'
                        surjective_idx = f'{en_idx}-{de_idx}'
                        new_parajoah_alignment_line.append(surjective_idx)
                        new_word_alignment_line.append(surjective_word)
                else:
                    #update new 
                    surjective_word = f'{surjective_word}<sep>{previous_de_word}'
                    surjective_idx = f'{surjective_idx}-{previous_de_idx}'
                    print("append", surjective_word, surjective_idx)
                    new_parajoah_alignment_line.append(surjective_idx)
                    new_word_alignment_line.append(surjective_word)
                    current_idx+=1
                    print("index", current_idx, last_idx)
                    if current_idx == last_idx:
                        surjective_word = f'{en_word}<sep>{de_word}'
                        surjective_idx = f'{en_idx}-{de_idx}'
                        new_parajoah_alignment_line.append(surjective_idx)
                        new_word_alignment_line.append(surjective_word)
                    #after appending, reset the surjective word
                    surjective_word, surjective_idx = en_word, en_idx
                    previous_de_word = de_word
                    previous_de_idx = de_idx
            
        '''          
        for pair in word_ende_dict:
            #check for one-to-one match
            if len(word_ende_dict[pair])==1:
                new_word_alignment = f"{pair[0]}<SEP>{word_ende_dict[pair][0][0]}"
                new_parajoah_alignment = f"{pair[1]}-{word_ende_dict[pair][0][1]}"
                new_word_alignment_line.append(new_word_alignment)
                new_parajoah_alignment_line.append(new_parajoah_alignment)
        '''
        all_new_word_alignments.append(new_word_alignment_line)
        all_new_parajoah_alignments.append(new_parajoah_alignment_line)
        previous_de_word, previous_de_idx = None, None
        surjective_word, surjective_idx = None, None
        iteration+=1
    return all_new_word_alignments, all_new_parajoah_alignments

def remove_stop_words(all_new_word_alignments, all_new_parajoah_alignments):
    new_word_alignment_no_stopwords, new_parajoah_alignment_no_stopwords = [], []
    for line_word, line_idx in zip(all_new_word_alignments, all_new_parajoah_alignments):
        new_word_line, new_idx_line = [], []
        for word, idx in zip(line_word, line_idx):
            en_word, _ = word.strip().split('<sep>')
            if en_word.lower() in stop_words or en_word.isdigit():
                continue
            else:
                new_word_line.append(word)
                new_idx_line.append(idx)
        new_word_alignment_no_stopwords.append(new_word_line)
        new_parajoah_alignment_no_stopwords.append(new_idx_line)
    return new_word_alignment_no_stopwords, new_parajoah_alignment_no_stopwords




if __name__ == "__main__":
    args = get_args()
    ip_list, iw_list = preprocessing(args.ip, args.iw)
    new_word_alignments, new_parajoah_alignments = align(ip_list, iw_list)
    new_word_alignments_no_stopwords, new_parajoah_alignments_no_stopwords = remove_stop_words(new_word_alignments, new_parajoah_alignments)
    
    with open(args.ow, 'w') as ow, open(args.op, 'w') as op:
        for word_line in new_word_alignments_no_stopwords:
            ow.write(" ".join(word_line) + "\n")

        for p_line in new_parajoah_alignments_no_stopwords :
            op.write(" ".join(p_line) + "\n")
    
    
            


    


        
                

            




