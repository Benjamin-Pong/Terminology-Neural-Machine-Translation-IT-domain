'''
This script prepares training and dev data for source and target languages
Output will be in custom data

Input
1. Outputs of ./clean_alignment.sh word alignment file (multiword terms accounted for)
2. idx alignment file
3. full corpus

Output: Train and dev splits for training , and also the final full training corpus after removing problematic instances
A) two types of augmentation
- frequency
- importance scoring
training
dev

B) full training corpos in mt/actual_training_corpus

'''
import argparse
import re
import string
import random
import json
import statistics
import torch
from fairseq.models.transformer import TransformerModel



def get_args():
    #inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--word_align", type=str, help="Path to word alignment file.")
    parser.add_argument("--idx_align", type=str, help='path to idx alignment file.')
    parser.add_argument("--corpora", type=str, help="path to the indomain or outdomain corpora")
    parser.add_argument("--freqDict", type = str, help="path to the frequency list")
    parser.add_argument("--aug_strat", type=str, help="augmentation strategy; no_aug, freq or importance")
    parser.add_argument("--threshold", type=float, help="probability threshold for terminology selection.")
    parser.add_argument("--amt_to_aug", type=float, help="the proportion of data to augment")
    parser.add_argument("--k", type=int, help="number of words per sentence to augment (freq)")
    
    #arguments for importance-aware augmentation
    parser.add_argument("--importance_corpora", type=str, help="path to importance corpora, which is the output of importance_scoring.py")
    parser.add_argument("--norm_threshold", type=int, help="top-k norms/tokens to choose for importance-augmentation")
    
    parser.add_argument("--model", type=str, help="name of pretrained model")
    #output
    parser.add_argument("--src_train", type = str, help= "dump out train data for src")
    parser.add_argument("--src_dev", type = str, help="dump out dev data for src")
    parser.add_argument("--tgt_train", type=str, help="dump out train data for tgt.")
    parser.add_argument("--tgt_dev", type=str, help="dump out dev data for tgt")
    parser.add_argument("--raw_training_corpora", type=str, help="raw training corpora post cleaning for the purpose of documentation.")
    
    return parser.parse_args()


class Augmentation:
    def __init__(self, raw_data=None, augmented_data=None):
        self.raw_data = raw_data #a string containing src ||| tgt
        self.augmented_data = augmented_data #tuple containing source, target

def preprocess(word_align, idx_align, corpora, freq):
    word_align_list, idx_align_list, corpora_list, freqDict= [], [], [], {}
    with open(word_align, 'r', encoding='utf-8') as wa, open(idx_align, 'r', encoding='utf-8') as ia, open(corpora, 'r', encoding='utf-8') as c, open(freq, 'r', encoding='utf-8') as fr:
        for wordline in wa:
            word_line_list = [wordline.strip().split()]
            word_align_list.append(word_line_list)
        
        for idxline in ia:
            idx_line_list = [idxline.strip().split()]
            idx_align_list.append(idx_line_list)
        
        for corpora_line in c:
            corpora_list.append(corpora_line.strip())
        
        for line in fr:
            #print(line)
            word, prob = line.strip().split('\t')
            freqDict[word] = prob
        
    word_align_list_ = []
    idx_align_list_ = []
    corpora_list_ = []

    # This control structure is to remove the empty lines
    for wordline, idxline, corpora_line in zip(word_align_list, idx_align_list, corpora_list):
        #print(wordline, idxline, corpora_line)
        if wordline == [] or idxline == []:
            #print("empty_line")
            continue
    
        else:
            word_align_list_.append(wordline)
            idx_align_list_.append(idxline)
            corpora_list_.append(corpora_line)

    return word_align_list_, idx_align_list_, corpora_list_, freqDict
    
def augment_freq(word_align, idx_align, corpora, freqDict, threshold, proportion_to_aug, k):
    '''
    Augments the data based on frequency statistics - modified to augment up to 2 words per sentence
    '''
    num_augmented_src_words = 0
    num_augmented_sents=0
    random.seed(42)
    freq_augmented_corpora = []
    count = 0
    error_lines = 0
    for word_line, idx_line, corpora_line in zip(word_align, idx_align, corpora):
        if random.random()< proportion_to_aug and num_augmented_src_words != 160000: #then augment
            num_augmented_sents+=1
            try:
                en_sent, de_sent = corpora_line.strip().split(" ||| ")
                en_sent_aug = en_sent.split() #has to be a list since the indices from the word aligner corresponds to token indices
                
                # Collect all candidate words for augmentation
                candidates = []
                for word_pair_, idx_pair_ in zip(word_line, idx_line):
                    for word_pair, idx_pair in zip(word_pair_, idx_pair_):
                        en_word, de_word = word_pair.strip().split('<sep>')
                        en_idx, de_idx = idx_pair.strip().split('-')
                        
                        if re.search(r'<w>', en_word): #check if multiword
                            en_word_split = en_word.split('<w>')
                            should_augment = False
                            for sub_word in en_word_split:
                                if sub_word == "":
                                    continue
                                if float(freqDict[sub_word.lower().strip(string.punctuation)]) < threshold:
                                    should_augment = True
                                    break
                            if should_augment:
                                candidates.append((en_word, de_word, en_idx, True))  # True = multiword
                        else: #single word
                            if en_word == "" or en_word not in freqDict:
                                continue
                            if float(freqDict[en_word.lower().strip(string.punctuation)]) < threshold:
                                candidates.append((en_word, de_word, en_idx, False))  # False = single word
                
                # If we have candidates, randomly select up to 2 to augment
                if candidates:
                    # Select up to 2 candidates (or all if fewer than 2)
                    num_to_select = min(k, len(candidates))
                    selected_candidates = random.sample(candidates, num_to_select)
                    
                    for en_word, de_word, en_idx, is_multiword in selected_candidates:
                        if is_multiword:
                            en_word_split = en_word.split('<w>')
                            start_idx, end_idx = int(en_idx.strip()[0]), int(en_idx.strip()[-1])
                            en_sent_aug[start_idx] = f'<src>{en_word_split[0]}'
                            en_sent_aug[end_idx] = f'{en_word_split[-1]}<tgt>{de_word}'
                        else:
                            en_sent_aug[int(en_idx)] = f'<src>{en_word}<tgt>{de_word}'
                        
                        num_augmented_src_words += 1
                    
                    # Only print when we actually augmented something
                    print(" ".join(en_sent_aug))
                    count+=1
                    print(count)
                
                    freq_augmented_corpora.append(Augmentation(raw_data=corpora_line.strip(),augmented_data = (" ".join(en_sent_aug), de_sent) ))
            except Exception as e:
                print(f'error:{e}')
                error_lines += 1
        else:
            #do not augment, and keep the original data format
            try:
                en_sent, de_sent = corpora_line.strip().split(" ||| ")
                freq_augmented_corpora.append(Augmentation(raw_data=corpora_line.strip(), augmented_data = (en_sent, de_sent) ))
            except Exception as e:
                print(f'error{e}')
                error_lines += 1
            
    print(count, error_lines)
    return freq_augmented_corpora, num_augmented_src_words, num_augmented_sents

def combine_scores(importance_scores, pharoah):
    '''
    This method creates the a new importance scores list by accounting for multiword. 
    Output should be a list of importance scores whose length is equal to the len of the pharoah list.
    '''
    importance_scores_multiword = []
    for alignment in pharoah:
        en_indices, _ = alignment.split('-')
        #check if it is multiwords. 
        if "~" in en_indices: #then it is multiword
            multiword_indices = en_indices.split("~") #list
            importance_scores_multiword.append(max([importance_scores[int(idx)] for idx in multiword_indices]))
        else:
            importance_scores_multiword.append(importance_scores[int(en_indices)])
        
    return importance_scores_multiword
            
            
def augment_importance(importance_corpora, norm_threshold):
    importance_augmented_corpora = []
    number_of_augmented_words=0
    '''
    Input: a json list of dictionaries, which is the output of  There are two types of dictionaries. One with only 2 key-value pair. These do not requirement augmentation. The other has 4 key-value pairs.These need to be augmented.
    
    Output: List Augmentation objects
    '''
    count=0
    with open(importance_corpora, 'r', encoding='utf-8') as f:
        importance_data = json.load(f)
    for data_instance in importance_data:
        try:
            src_sent, tgt_sent = data_instance['src_sent'], data_instance['tgt_sent']
            count+=1
            print(count)
    
            if len(data_instance)==2: #no augmentation required, so use original data
                importance_augmented_corpora.append(Augmentation(raw_data=f'{src_sent} ||| {tgt_sent}' , augmented_data=(src_sent, tgt_sent)))
    
            else:
                print("more")
                pharoah = data_instance['parajoah']#list
                if not pharoah:
                    continue
                tgt_tokens = tgt_sent.strip().split() #list
                src_tokens = data_instance['word_list']
                src_tokens_aug = src_tokens[:]
                importance_scores = data_instance['norm_list']#list
                #print("source", src_tokens)
                #print("target", tgt_tokens)
                #print("pharoah", pharoah)
                #print("norm per source tokens", importance_scores)
                
                #Decide on the scoring threshold?
                    #First, combine the scores first for multiword tokens - use average
                
                multiword_norms = combine_scores(importance_scores, pharoah)
                # Get the top-k norms and their indices
                top_k_indices = []
                #print(multiword_norms)
                if len(multiword_norms) >=norm_threshold:
                    
                    top_k_norms, top_k_indices_tensor = torch.topk(torch.tensor(multiword_norms, dtype=torch.float32), norm_threshold ) # k = number of top tokens to select
                else: #if number of norms is less than k
                    
                    top_k_norms, top_k_indices_tensor = torch.topk(torch.tensor(multiword_norms, dtype=torch.float32), norm_threshold -1)
                top_k_indices = top_k_indices_tensor.tolist()
                #print(top_k_indices)
                for idx in top_k_indices:
                    number_of_augmented_words +=1
                    pharoah_alignment_chosen = pharoah[idx]
                    #print("top_k pharoah alignments", pharoah_alignment_chosen)
                    #check if it is a multiword alignment
                    if '~' in pharoah_alignment_chosen:
                        en_indices, de_idx = pharoah_alignment_chosen.split('-')
                        all_en_indices = en_indices.split('~')
                        start_en_idx, end_en_idx = int(all_en_indices[0].strip()), int(all_en_indices[-1].strip())
                        src_tokens_aug[start_en_idx] = f'<src>{src_tokens[start_en_idx]}'
                        src_tokens_aug[end_en_idx]=f'{src_tokens[end_en_idx]}<tgt>{tgt_tokens[int(de_idx)].strip(string.punctuation)}'
                    else:
                        en_idx, de_idx = pharoah_alignment_chosen.split('-')
                        #print(en_idx, de_idx)
                        src_tokens_aug[int(en_idx)] = f'<src>{src_tokens[int(en_idx)]}<tgt>{tgt_tokens[int(de_idx)]}'
                        
                src_augmented_sent = " ".join(src_tokens_aug)
                print(src_augmented_sent)
                importance_augmented_corpora.append(Augmentation(raw_data=f'{src_sent} ||| {tgt_sent}' , augmented_data=(src_augmented_sent, tgt_sent)))
            
        except Exception as e:
            print(e)
    #print(len(importance_augmented_corpora))
    return importance_augmented_corpora, number_of_augmented_words
    
def augment_hybrid(word_align, idx_align, corpora, freqDict, threshold, proportion_to_aug, k, model):
    """
    Hybrid augmentation: Filter by frequency, then select top-k by importance score
    
    Args:
        word_align: Word alignment data
        idx_align: Index alignment data  
        corpora: Corpus data
        freqDict: Frequency dictionary
        threshold: Frequency threshold for filtering
        proportion_to_aug: Proportion of sentences to augment
        k: Number of top words to select per sentence
        model: Pre-trained model for computing importance scores
    """
    import torch
    import string
    
    html_entities = {'&quot;', '&amp;', '&lt;', '&gt;', '&nbsp;', '&apos;'}
    num_augmented_src_words = 0
    num_augmented_sents = 0
    random.seed(42)
    hybrid_augmented_corpora = []
    count = 0
    error_lines = 0
    
    def compute_norm(vector):
        """Compute L2 norm of vector"""
        return torch.linalg.norm(vector)
    
    def compute_max_pooling(word_bpe_hidden_states):
        """Compute max pooling over BPE hidden states"""
        word_bpe_hidden_states_ = torch.stack(word_bpe_hidden_states)
        max_pooled_hidden_state = torch.max(word_bpe_hidden_states_, dim=0)[0]
        return max_pooled_hidden_state
    
    for word_line, idx_line, corpora_line in zip(word_align, idx_align, corpora):
        if random.random() < proportion_to_aug and num_augmented_src_words != 60000:
            num_augmented_sents += 1
            try:
                en_sent, de_sent = corpora_line.strip().split(" ||| ")
                en_sent_aug = en_sent.split()
                
                # Step 1: Get encoder hidden states and reconstruct words from BPE
                with torch.no_grad():
                    tokens = model.encode(en_sent.strip())
                    token_ids = tokens
                    bpe_tokens = [model.task.source_dictionary[id] for id in token_ids]
                    
                    if len(bpe_tokens) > 800:  # Skip very long sentences
                        continue
                    
                    # Get encoder hidden states
                    encoder_out = model.models[0].encoder(tokens.unsqueeze(0))
                    encoder_hidden_states = encoder_out['encoder_out'][0]
                
                # Step 2: Reconstruct words from BPE tokens (your exact logic)
                need_to_reconstruct = False
                word_list, norm_list = [], []
                word_to_reconstruct, hidden_state_to_reconstruct = [], []
                
                for bpe_token, encoder_hidden_state in zip(bpe_tokens, encoder_hidden_states):
                    # Skip </s> tokens and punctuation
                    if bpe_token == '</s>' or bpe_token in string.punctuation or bpe_token in html_entities:
                        continue
                    
                    if bpe_token.endswith("@@"):
                        need_to_reconstruct = True
                        word_to_reconstruct.append(bpe_token[:-2])
                        hidden_state_to_reconstruct.append(encoder_hidden_state)
                    else:
                        if need_to_reconstruct:
                            # Multi-subword case
                            word_to_reconstruct.append(bpe_token)
                            hidden_state_to_reconstruct.append(encoder_hidden_state)
                            need_to_reconstruct = False
                            
                            # Reconstruct the word
                            new_word = "".join(word_to_reconstruct)
                            new_hidden_state = compute_max_pooling(hidden_state_to_reconstruct)
                            norm_new_hidden_state = compute_norm(new_hidden_state)
                            
                            word_list.append(new_word)
                            norm_list.append(norm_new_hidden_state.tolist())
                            word_to_reconstruct, hidden_state_to_reconstruct = [], []
                        else:
                            # Single word case
                            word_list.append(bpe_token)
                            norm_ = compute_norm(encoder_hidden_state)
                            norm_list.append(norm_.tolist())
                
                # Step 3: Handle hyphenated tokens (your exact logic)
                word_list_recon, norm_list_recon = [], []
                i = 0
                if "@-@" in word_list:
                    while i < len(word_list):
                        if i+2 < len(word_list) and word_list[i+1] == '@-@':
                            word_list_recon.append(word_list[i] + '-' + word_list[i+2])
                            norm_list_recon.append(max(float(norm_list[i]), float(norm_list[i+1]), float(norm_list[i+2])))
                            i += 3
                        else:
                            word_list_recon.append(word_list[i])
                            norm_list_recon.append(norm_list[i])
                            i += 1
                else:
                    word_list_recon = word_list
                    norm_list_recon = norm_list
                
                # Step 4: Process alignments with reconstructed words
                candidates = []
                for word_pair_, idx_pair_ in zip(word_line, idx_line):
                    for word_pair, idx_pair in zip(word_pair_, idx_pair_):
                        en_word, de_word = word_pair.strip().split('<sep>')
                        en_idx = idx_pair.strip().split('-')[0]
                        
                        # Find corresponding word in reconstructed list and do frequency check
                        try:
                            if '~' in en_idx:  # Multiword alignment (like "0~1~2")
                                indices = list(map(int, en_idx.split('~')))
                                # Reconstruct the multiword from word_list_recon
                                multiword_parts = [word_list_recon[i] for i in indices]
                                word_text = "".join(multiword_parts)  # or use space: " ".join()
                                
                                # Check frequency for each part
                                should_augment = False
                                for part in multiword_parts:
                                    clean_part = part.lower().strip(string.punctuation)
                                    if clean_part in freqDict and float(freqDict[clean_part]) < threshold:
                                        should_augment = True
                                        break
                                
                                if should_augment:
                                    # Get importance score (max of constituent parts)
                                    word_norm = max([float(norm_list_recon[i]) for i in indices])
                                    candidates.append({
                                        'word': word_text,
                                        'de_word': de_word,
                                        'indices': indices,
                                        'importance': word_norm,
                                        'is_multiword': True
                                    })
                            else:
                                # Single word alignment
                                idx = int(en_idx)
                                if idx >= len(word_list_recon):
                                    continue
                                    
                                word_text = word_list_recon[idx]
                                clean_word = word_text.lower().strip(string.punctuation)
                                
                                # Frequency check
                                if clean_word in freqDict and float(freqDict[clean_word]) < threshold:
                                    word_norm = float(norm_list_recon[idx])
                                    candidates.append({
                                        'word': word_text,
                                        'de_word': de_word,
                                        'indices': [idx],
                                        'importance': word_norm,
                                        'is_multiword': False
                                    })
                        except (IndexError, ValueError) as e:
                            print(f"Alignment error: {e}")
                            continue
                
                # Step 5: Select top-k candidates by importance score
                if candidates:
                    # Sort by importance score (descending)
                    candidates.sort(key=lambda x: x['importance'], reverse=True)
                    
                    # Select top-k candidates
                    num_to_select = min(k, len(candidates))
                    selected_candidates = candidates[:num_to_select]
                    
                    # Step 6: Apply augmentation to the original tokens
                    for cand in selected_candidates:
                        if cand['is_multiword']:
                            # For multiword, we need to map back to original token indices
                            # This is tricky because word_list_recon indices != original token indices
                            # We'll use the alignment information from en_word in the original data
                            
                            # Find the original alignment data for this candidate
                            for word_pair_, idx_pair_ in zip(word_line, idx_line):
                                for word_pair, idx_pair in zip(word_pair_, idx_pair_):
                                    en_word_orig, de_word_orig = word_pair.strip().split('<sep>')
                                    en_idx_orig = idx_pair.strip().split('-')[0]
                                    
                                    # Match by target word (more reliable than source)
                                    if de_word_orig == cand['de_word']:
                                        if '<w>' in en_word_orig:  # Multiword in original format
                                            en_word_split = en_word_orig.split('<w>')
                                            start_idx, end_idx = int(en_idx_orig.strip()[0]), int(en_idx_orig.strip()[-1])
                                            en_sent_aug[start_idx] = f'<src>{en_word_split[0]}'
                                            en_sent_aug[end_idx] = f'{en_word_split[-1]}<tgt>{de_word_orig}'
                                        break
                        else:
                            # Single word - also need to map back to original indices
                            for word_pair_, idx_pair_ in zip(word_line, idx_line):
                                for word_pair, idx_pair in zip(word_pair_, idx_pair_):
                                    en_word_orig, de_word_orig = word_pair.strip().split('<sep>')
                                    en_idx_orig = idx_pair.strip().split('-')[0]
                                    
                                    if de_word_orig == cand['de_word'] and '~' not in en_idx_orig:
                                        en_sent_aug[int(en_idx_orig)] = f'<src>{en_word_orig}<tgt>{de_word_orig}'
                                        break
                        
                        num_augmented_src_words += 1
                    
                    print(" ".join(en_sent_aug))
                    count += 1
                    print(f"Sentence {count} - Selected {len(selected_candidates)} words by importance")
                    
                    hybrid_augmented_corpora.append(Augmentation(
                        raw_data=corpora_line.strip(),
                        augmented_data=(" ".join(en_sent_aug), de_sent)
                    ))
                    
            except Exception as e:
                print(f'Error: {e}')
                error_lines += 1
        else:
            # Don't augment, keep original format
            try:
                en_sent, de_sent = corpora_line.strip().split(" ||| ")
                hybrid_augmented_corpora.append(Augmentation(
                    raw_data=corpora_line.strip(),
                    augmented_data=(en_sent, de_sent)
                ))
            except Exception as e:
                print(f'Error: {e}')
                error_lines += 1
    
    print(f"Processed: {count} augmented sentences, {error_lines} errors")
    print(f"Total augmented words: {num_augmented_src_words}")
    print(f"Total augmented sentences: {num_augmented_sents}")
    
    return hybrid_augmented_corpora, num_augmented_src_words, num_augmented_sents  
    
def  no_aug(corpora_list):
    output = []
    for corpora_line in corpora_list:
        try:
            src_sent, tgt_sent = corpora_line.strip().split(" ||| ")
            output.append(Augmentation(raw_data=corpora_line, augmented_data=(src_sent, tgt_sent)))
        except Exception as e:
            print(e)
    return output
        
        
def dump_out(freq_augmented_corpora, src_train, src_dev, tgt_train, tgt_dev, raw_corpora):
    random.shuffle(freq_augmented_corpora)
    #print(freq_augmented_corpora)
    
    train = freq_augmented_corpora[: int(0.9*len(freq_augmented_corpora))]
    dev = freq_augmented_corpora[int(0.9*len(freq_augmented_corpora)):]
    with open(src_train, 'w', encoding='utf-8') as st, open(src_dev, 'w', encoding='utf-8') as sd, open(tgt_train, 'w', encoding='utf-8') as tt, open(tgt_dev, 'w', encoding='utf-8') as td, open(raw_corpora, 'w', encoding='utf-8') as rc:
        for tr in train:
            src, tgt = tr.augmented_data[0], tr.augmented_data[1]
            st.write(src)
            st.write("\n")
            tt.write(tgt)
            tt.write("\n")
        
        for dv in dev:
            src, tgt = dv.augmented_data[0], dv.augmented_data[1]
            sd.write(src)
            sd.write("\n")
            td.write(tgt)
            td.write("\n")

        for data in freq_augmented_corpora:
            rc.write(data.raw_data)
            rc.write('\n')


if __name__ == "__main__":
    args = get_args()
    strategy = args.aug_strat
    if strategy == 'freq':
        print(strategy)
        word_align_list_, idx_align_list_, corpora_list_, freqDict = preprocess(args.word_align, args.idx_align, args.corpora, args.freqDict)
        if len(word_align_list_)==len(idx_align_list_)==len(corpora_list_):
            print ("True")
        else:
            print("something is wrong")
            quit()
        freq_augmented_corpora, num_words_augmented, num_augmented_sents = augment_freq(word_align_list_, idx_align_list_, corpora_list_, freqDict, args.threshold, args.amt_to_aug, args.k)
        print("total eventual corpora:", (len(freq_augmented_corpora)))
        print("number of augmented words:", num_words_augmented)
        print("number of augmented sents:", num_augmented_sents)
        #freq_augmented_corpora = augment_freq(word_align_list, idx_align_list, corpora_list, freqDict, args.threshold)
        dump_out(freq_augmented_corpora, args.src_train, args.src_dev, args.tgt_train, args.tgt_dev, args.raw_training_corpora)
    elif strategy == "no_aug":
        print(strategy)
        word_align_list_, idx_align_list_, corpora_list_, freqDict = preprocess(args.word_align, args.idx_align, args.corpora, args.freqDict)
        no_aug_corpora = no_aug(corpora_list_)
        print(len(no_aug_corpora))
        dump_out(no_aug_corpora, args.src_train, args.src_dev, args.tgt_train, args.tgt_dev, args.raw_training_corpora )
        
    elif strategy=="importance": #importance augmentation
        print(strategy)
        importance_augmented_corpora,number_of_augmented_words =augment_importance(args.importance_corpora, args.norm_threshold)
        #print(importance_augmented_corpora)
        dump_out(importance_augmented_corpora,args.src_train, args.src_dev, args.tgt_train, args.tgt_dev, args.raw_training_corpora)
        
        print("size of corpora", len(importance_augmented_corpora))
        print("augmented_words", number_of_augmented_words)
    else: #hybrid
        print(strategy)
  
        model = TransformerModel.from_pretrained(
        args.model,
        checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
        tokenizer='moses',
        bpe='fastbpe'
    )
        word_align, idx_align, corpora, freqDict = preprocess(args.word_align, args.idx_align, args.corpora, args.freqDict)
        hybrid_augmented_corpora, num_augmented_src_words, num_augmented_sents= augment_hybrid(word_align, idx_align, corpora, freqDict, args.threshold, args.amt_to_aug, args.k, model)
        
        print("size of training corpora", len(hybrid_augmented_corpora))
        print("num of augmented words", num_augmented_src_words)
        print("num of augmented sents", num_augmented_sents)
        
        dump_out(hybrid_augmented_corpora,args.src_train, args.src_dev, args.tgt_train, args.tgt_dev, args.raw_training_corpora)
        
        
        
        
        
        




