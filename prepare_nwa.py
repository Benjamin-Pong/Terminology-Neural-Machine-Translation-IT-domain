import argparse

'''
This script takes the top 50000 ced scored data that has already been sorted, and converts it into a format for neural word alignment. 
'''
def get_args():
    parser= argparse.ArgumentParser()
    parser.add_argument("--new_data", type=str, help="Path to the ced scored data.")
    parser.add_argument("--nwa_indata", type=str, help="Path to the nwa in-domain data.")
    parser.add_argument("--nwa_outdata", type=str, help="Path to the nwa out-domain data.")
    return parser.parse_args()

def convert_to_nwa_format(new_data, nwa_indata, nwa_outdata):
    all_data = []
    count=0
    with open (new_data, 'r', encoding='utf-8') as f:
        for line in f:
            #print(line)
            ced_score, source_sentence, target_sentence = line.strip().split('\t')
            all_data.append((source_sentence.strip(), target_sentence.strip()))
            # Convert the source and target sentences to the nwa format

    with open(nwa_indata, 'w', encoding='utf-8') as nwa_file_in, open(nwa_outdata, 'w', encoding='utf-8') as nwa_file_out:    
        for data in all_data[0:2200000]:
            source_sentence, target_sentence = data[0], data[1]
            # Write to nwa format
            nwa_format = f"{source_sentence} ||| {target_sentence}\n"
            nwa_file_in.write(nwa_format)
            count+=1
            print(nwa_format)
            print(count)
        
        #get last 90000 
        for data in all_data[4000000:6200000]:
            source_sentence, target_sentence = data[0], data[1]
            # Write to nwa format
            nwa_format = f"{source_sentence} ||| {target_sentence}\n"
            nwa_file_out.write(nwa_format)
            count+=1
            print(nwa_format)
            print(count)
        
                

if __name__ == "__main__":
    args = get_args()
    convert_to_nwa_format(args.new_data, args.nwa_indata, args.nwa_outdata)
    