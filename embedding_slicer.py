import argparse
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

parser_debug = parser.add_argument_group('debug options')
parser_debug.add_argument('--device',choices=['auto','cpu','gpu'],default='auto')

parser_data = parser.add_argument_group('data options')
parser_data.add_argument('--data',type=str)
parser_data.add_argument('--save_path',type=str)

parser_model=parser.add_argument_group('model options')
parser_model.add_argument('--num_lines',type=int)

#parser_model.add_argument('--batch_size',type=int)

args=parser.parse_args()

import glob
import os
import math
import random
import string
import sys
import torch
import datetime

#set device
if args.device=='gpu' or (torch.cuda.is_available() and args.device=='auto'):
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
else:
    device = torch.device('cpu')

print('\n')
print('device=',device)


#retrieve valid files 
file_list=glob.glob(args.data+'/*'+'.bert')
filename_list=[]

#display basename of files to user
for file in file_list:
	filename=os.path.basename(file)
	short_filename=filename.split('6way.')[1].lstrip().split('.')[0]
	filename_list.append(short_filename)

print('Files to be sliced:',filename_list)


for file in file_list:
	
	print('\n')
	print('Current file',os.path.basename(file))
	print('Tensor slicing in progress...')
	print('\n')
	
	input_tensor=torch.load(file)
	
	file_lengths=list(input_tensor.shape)[0]
	
	print('File lengths:',file_lengths)
	print('Tensor shapes:',list(input_tensor.shape))
	
	#set seed so each file splitting is the same 
	random.seed(50)
	random_num_list=random.sample(range(0,file_lengths),args.num_lines)
	#print('Random number list, type: list', random_num_list)
	tensor_indicies=torch.LongTensor(random_num_list)
	#print('Random number list, type: LongTensor',tensor_indicies)
	output_tensor=torch.index_select(input_tensor,0,tensor_indicies)
	print('Output Tensor:',output_tensor)
	print('\n')
	print('Saving Tensors...')
	save_loc_path=args.save_path+'/'+os.path.basename(file)+'.sliced_version'
	torch.save(output_tensor,save_loc_path)
	print('Saving complete')

##load only the first file 
#embedding_file1=torch.load(file_list[0])
#file_lengths=list(embedding_file1.shape)[0]

#print('File lengths:',file_lengths)
#print('Tensor shapes:',list(embedding_file1.shape))


##calculate the max value for the random slicing indexs  
#random_num_list=random.sample(range(1,file_lengths),args.num_lines)

#print('Length of random list:',len(random_num_list))

##go through each file in the folder and preform the slicing at the random indexs 
#for file in file_list:
	#print('\n')
	#print('Current file',os.path.basename(file))
	#print('Tensor slicing in progress...')
	#print('\n')
	
	#tensor_list=[]
	#line_counter=0
	#embedding_file=torch.load(file)
	
	#for slice_index in random_num_list:
		
		#line_counter+=1
		
		#print('Line number:',line_counter)
		#sliced_tensor_piece=torch.narrow(embedding_file,0,slice_index,1)
		#tensor_list.append(sliced_tensor_piece)
		#print('Sliced tensor shape:',sliced_tensor_piece.shape) #debugging
		#print('Adding to tensor list') #debugging
		#print('The tensor list now contains',len(tensor_list),'tensors') #debugging
		#print('\n') #debugging
		
		##if line_counter%args.batch_size==0 or line_counter==len(random_num_list):
			
			##if line_counter%args.batch_size==0:
				##print('Batch size reached')
			
		#if line_counter==len(random_num_list):
			#print('End of file reached') #debugging

			
			##print('Final tensor current shape:',sliced_tensor_final.shape)
			
			##print('\n')
		
			
		
	#print('Concatinating to final tensor')
	#sliced_tensor_final=torch.cat(tensor_list)
	
	
	#print('Slice index:',random_num_list.index(slice_index)) #debugging
	#print('These should be equal',len(random_num_list)-1,':',random_num_list.index(slice_index))
	#print('\n') #debugging
	#print('Final tensor shape:',sliced_tensor_final.shape)
	
	##saving the sliced versions
	
	

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
