import argparse
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

parser_debug = parser.add_argument_group('debug options')
parser_debug.add_argument('--device',choices=['auto','cpu','gpu'],default='auto')

parser_data = parser.add_argument_group('data options')
parser_data.add_argument('--data',type=str)
parser_data.add_argument('--storage_type',type=str, default='folder')
parser_data.add_argument('--max_file_length', type=int)

parser_display = parser.add_argument_group('display options')
parser_display.add_argument('--print_tensor',action='store_true')
parser_display.add_argument('--print_cuda_stats',action='store_true')


parser_model = parser.add_argument_group('model options')
parser_model.add_argument('--model',choices=['bert'],default='bert')
parser_model.add_argument('--batch_size',type=int)

args = parser.parse_args()

if args.batch_size is None:
	raise ValueError('Must specify batch size, --batch_size=?')

if args.max_file_length is None:
	raise ValueError('Must specify the max number of lines to encode, --max_file_length=?')

# supress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# load modules
import datetime
import glob
import os
import math
import random
import string
import sys
import time
import gc
import unicodedata
from unidecode import unidecode



# import the deep learning libraries
import torch
import transformers



# set device to cpu/gpu
if args.device=='gpu' or (torch.cuda.is_available() and args.device=='auto'):
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
else:
    device = torch.device('cpu')
print('device=',device)
1
# import the training data
BOL = '\x00'
EOL = '\x01'
OOV = '\x02'
vocabulary = string.ascii_letters
vocabulary += " .,;'" + '1234567890:-/#$%' + OOV + BOL + EOL

def unicode_to_ascii(s):
    '''
    Removes diacritics from unicode characters.
    See: https://stackoverflow.com/a/518232/2809427
    '''
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in vocabulary
    )

def format_line(line):
    line = unidecode(line)
    return line

# load the model
#checkpoint_name = 'bert-base-uncased'
checkpoint_name = 'bert-base-multilingual-uncased'
tokenizer = transformers.BertTokenizer.from_pretrained(checkpoint_name)
bert = transformers.BertModel.from_pretrained(checkpoint_name)

def bert_encodings(ss):
    input_ids=[]
    attention_masks=[]
    for line in ss:
        encoded_dict=tokenizer.encode_plus(line,add_special_tokens=True,max_length=64,pad_to_max_length=True,return_attention_mask=True,return_tensors='pt',)
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids=torch.cat(input_ids,dim=0)
    attention_masks=torch.cat(attention_masks,dim=0)
    return input_ids
	

def data_parser():
	for file in os.listdir(args.data):
		with open(args.data+'/'+file,'r') as f_in:
			line_counter=0
			lines=[]
			tensor_list=[]
			for line in f_in:
				with torch.no_grad():
					line_counter+=1
					if line_counter%(args.max_file_length/100) == 0:
						print(str(line_counter*100/args.max_file_length)+'Percent completed of file:'+file)
						#print(round(torch.cuda.memory_allocated/1000000000),6)
						if args.print_cuda_stats:
							print(torch.cuda.memory_summary(device=None, abbreviated=True))
					if len(lines)< args.batch_size:
						lines.append(line.strip())	
					if len(lines)== args.batch_size or line_counter==args.max_file_length:
						line_tensor = bert_encodings(lines)
						last_layer,embedding = bert(line_tensor) 
						embedding = torch.mean(last_layer,dim=1)
						embedding_cpu=embedding.to(torch.device('cpu'))
						tensor_list.append(embedding_cpu)
						lines=[]
					if line_counter==args.max_file_length:
						break
							
			concat_tensor=torch.cat(tensor_list)
			file_name=args.data+'/'+os.path.basename(file)+'.bert'
			torch.save(concat_tensor,file_name)	
			if args.print_tensor:
				print(concat_tensor.shape)

data_parser()






