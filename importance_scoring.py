


import torch
import argparse
import string
from fairseq.models.transformer import TransformerModel
import json
import random
'''
This script computes the importance scores of each token in the source data, and outputs a json file of all data instances of the following format.

Note that we are only computing the importance scores of a portion of the data, to save time, since computing importance scores entail augmentation, and we do not want to introduce too much noise..

Hence, the output would be a json list of dictionaries.

There are two types of dictionaries:

1. With importance scoring
[{'src_sent':str, 'tgt_sent':str, 'parajoah': str, tokens:List[string], l2_norm: List[float]


                }]
                
2. Without importance scoring

[{'src_sent':str, 'tgt_sent:str}......


]
                
Note that at this stage, we have not computed the importance scoring for multiwords yet!


'''

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="model name", required=True)
    parser.add_argument('--parallel_data', type=str, help="path to multiword-neural-aligned parallel data for importance scoring", required=False)
    parser.add_argument('--parajoah', type=str, help='path to the multiword-neural-aligned parajoah data', required=False)
    parser.add_argument('--output', type=str, help='path to the importance scores per token')
    parser.add_argument('--amt_to_aug', type=float, help = 'amount of sentences to do importance scoring for')
    return parser.parse_args()

def compute_norm(vector):
    '''
    Input: a tensor with dimensions [1 x n] where n>1
    Output: a tensor with dimension[1x1]
    '''
    return torch.linalg.norm(vector)
    

def compute_max_pooling(word_bpe_hidden_states):
    '''
    Compute the aggregation of tensors t
    word_bpe_hidden_states is a list of tensors 
    Need to convert it to a tensor before computing max pooling
    '''
    word_bpe_hidden_states_ = torch.stack(word_bpe_hidden_states)
    max_pooled_hidden_state = torch.max(word_bpe_hidden_states_, dim=0)[0]
    return max_pooled_hidden_state
def get_self_attention_importance(self_attentions, word_list_recon):
    """Returns a LIST of importance scores aligned with word_list_recon"""
    # self_attentions shape: [num_layers, heads, src_len, src_len]
    num_layers, num_heads, src_len, _ = self_attentions.shape
    
    # Average across layers and heads
    avg_attention = self_attentions.mean(dim=(0, 1))  # [src_len, src_len]
    
    # Importance = sum of how much others attend to each word
    importance_scores = avg_attention.sum(dim=0)  # [src_len]
    
    # Normalize to [0, 1]
    importance_scores = (importance_scores - importance_scores.min()) / \
                       (importance_scores.max() - importance_scores.min() + 1e-9)
    
    return importance_scores.tolist()  # Convert to Python list
if __name__ == "__main__":
    html_entities = {'&quot;', '&amp;', '&lt;', '&gt;', '&nbsp;', '&apos;'}
    random.seed(42)
    
    args = get_args()
# Load a pre-trained Fairseq model
    model = TransformerModel.from_pretrained(
    args.model,
    checkpoint_file='model1.pt:model2.pt:model3.pt:model4.pt',
    tokenizer='moses',
    bpe='fastbpe'
)

    # Ensure the model is in evaluation mode
    model.eval()
    output = []
    # Tokenize input text
    #count = 0
    instance = 0
    number_of_sents_with_importance_scoring = 0
    with open(args.parallel_data, 'r') as pd, open(args.parajoah) as p:
        for line, para in zip(pd,p):
            instance+=1
            print(instance)
            print(line, para)
            '''
            count +=1
            if count > 20:
                break
            '''
            data_instance = {}
            para_ = para.strip().split()
            line_ = line.strip()
            try:
                src_sent, tgt_sent = line_.split(' ||| ')
                data_instance['src_sent'] = src_sent.strip()
                data_instance['tgt_sent'] = tgt_sent.strip()
                
                if random.random()<args.amt_to_aug: #then compute the importance scoring
                    number_of_sents_with_importance_scoring+=1
                    data_instance['parajoah'] = para_
                    tokens = model.encode(src_sent.strip())
                    #print(tokens)
                    token_ids = tokens
                    bpe_tokens = [model.task.source_dictionary[id] for id in token_ids]
                    print("BPE Tokens:", bpe_tokens)
                    if len(bpe_tokens) > 800:
                        continue
                
                    # Forward pass through the model
                    with torch.no_grad():
                        # Get the encoder hidden states
                        encoder_out = model.models[0].encoder(tokens.unsqueeze(0))  # Add batch dimension
                        encoder_hidden_states = encoder_out['encoder_out']  
                    
                    #align the encoder hidden states to each BPE tokens
                    need_to_reconstruct = False
                    word_list, norm_list = [],[]
                    word_to_reconstruct, hidden_state_to_reconstruct  = [], []
                    for bpe_token, encoder_hidden_state in zip(bpe_tokens, encoder_hidden_states[0]):
                        #skip the </s> tokens,, and tokenized punctuations 
                        if bpe_token == '</s>' or bpe_token in string.punctuation or bpe_token in html_entities: #string.punctuation is a string
                            continue
                        else:
                            if bpe_token.endswith("@@"):
                                #print("bpe ends with @@")
                                need_to_reconstruct = True
                                word_to_reconstruct.append(bpe_token[:-2])
                                
                            else:
                                #just because a bpe_token does not end in @@, it does not mean that it is not a subword. It could be the last piece of the subword. Hence, check if the flag is True
                                #multi-subword case
                                if need_to_reconstruct == True:
                                    word_to_reconstruct.append(bpe_token)
                                    hidden_state_to_reconstruct.append(encoder_hidden_state)
                                    #marks the end of the reconstructed word
                                    need_to_reconstruct = False
                                    '''
                                    Need to compute the pooling
                                    Need to join the bpe_token
                                    Need to reset the lists for the next token
                                    '''
                                    new_word = "".join(word_to_reconstruct)
                                    new_hidden_state = compute_max_pooling(hidden_state_to_reconstruct) #[1 x 1024]
                                    norm_new_hidden_state = compute_norm(new_hidden_state)
                                    word_to_reconstruct, hidden_state_to_reconstruct  = [], []
                                    #print(f'new_word:{new_word}')
                                    #print(f'new_hidden_states:{new_hidden_state}')
                                    #print(f'norm_new_hidden_states:{norm_new_hidden_state}')
                                    word_list.append(new_word)
                                    norm_list.append(norm_new_hidden_state.tolist())
                                
                                else:
                                    #single word case (no subwords)
                                    #free to dump out the bpe_token  encoder_hidden_state correspondence.
                                    print(f'{bpe_token}\t{encoder_hidden_state}')
                                    word_list.append(bpe_token)
                                    norm_ = compute_norm(encoder_hidden_state)
                                    norm_list.append(norm_.tolist()) #convert t
                    
                    #logic to merge hyphenated tokens
                    word_list_recon, norm_list_recon = [],[] 
                    i = 0
                    if "@-@" in word_list:
                        while i < len(word_list):
                            if i+2 < len(word_list) and word_list[i+1]=='@-@':
                                word_list_recon.append(word_list[i]+'-'+word_list[i+2])
                                norm_list_recon.append(max(float(norm_list[i]), float(norm_list[i+1]), float(norm_list[i+2]))) #decision to select the max norm when combining hyphenated tokens
                                i+=3
                            else:
                                word_list_recon.append(word_list[i])
                                norm_list_recon.append(norm_list[i])
                                i+=1
                        data_instance['word_list']= word_list_recon
                        #print(word_list)
                        data_instance['norm_list']= norm_list_recon
                        print(data_instance)
        
                    else:
                        data_instance['word_list']= word_list
                        #print(word_list)
                        data_instance['norm_list']= norm_list
                       # print(norm_list)
                        print(data_instance)
                
                #else, just append the data_instance
                    
                output.append(data_instance)
            
            except Exception as e:
                print("Data error")
    
    
    with open(args.output, 'w', encoding='utf-8') as out:
        json.dump(output, out, indent=2)
    
    print(len(output)) #should be 2000000
    print(number_of_sents_with_importance_scoring) #should be around 200k
  
