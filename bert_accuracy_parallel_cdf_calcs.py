import argparse
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')


parser_data = parser.add_argument_group('data options')
parser_data.add_argument('--data_sliced',type=str)
parser_data.add_argument('--data_full',type=str)
parser_data.add_argument('--langs','--list', nargs='+',help='<Required> Set flag', required=True)
parser_data.add_argument('--save_outputs',action='store_true')
parser_data.add_argument('--slice_num',type=str)
parser_data.add_argument('--save_path',type=str)

parser_print=parser.add_argument_group('print options')
parser_print.add_argument('--file_names',action='store_true')
parser_print.add_argument('--print_line_num',action='store_true')
parser_print.add_argument('--print_all_data',action='store_true')

parser_model=parser.add_argument_group('model options')
parser_model.add_argument('--n_smallest_dists',type=int)
parser_model.add_argument('--test_run',action='store_true')

args = parser.parse_args()

import torch
import os
import glob
from statistics import mean
import pandas as pd
from heapq import nsmallest
import pickle

print(args.data_sliced+'/*'+args.langs[0]+'.bert.sliced_version.'+args.slice_num+'.split')
print(args.data_full+'/*'+args.langs[1]+'.bert.sliced_version')
file1_name=glob.glob(args.data_sliced+'/*'+args.langs[0]+'.bert.sliced_version.'+args.slice_num+'.split')[0]
file2_name=glob.glob(args.data_full+'/*'+args.langs[1]+'.bert.sliced_version')[0]


print('\n')
print('Data and device information:')
device = torch.device('cpu')
print('Device used:',device)

file_list=[file1_name,file2_name]
file_list_output=[]


for file in file_list:
	filename=os.path.basename(file)
	short_filename=filename.split('6way.')[1].lstrip().split('.')[0]
	file_list_output.append(short_filename)



if args.file_names:
	print('Files: ',file_list_output)

def accuracy_calc():
	
	#load files:
	print('\n')
	print('Loading Files...')
	print('\n')
	file1=torch.load(file1_name)
	file2=torch.load(file2_name)
	file1_length=list(file1.shape)
	file2_length=list(file2.shape)
	
	#output information
	print('File1 size:')
	print(file1_length[0],'lines')
	print('File2 size:')
	print(file2_length[0],'lines')
	
	#initialize storage lists and indicies
	diag_dists=[]
	k_min_dists=[]
	accuracy=[]
	diag_found_in_k_min_dist_rate_list=[]
	embedding1_index=0
	line_counter=0
	
	
	#distance calculations loop
	print('\n')
	print('Calculating distances...')
	for embedding1 in file1:
		
		line_counter+=1
		dist_list=[]
		
		#halts for loop early for testing purposes 
		if args.test_run:
			if line_counter==100:
				break
		
		for embedding2 in file2:
			
			#calculate each distance for line in file2
			dist=torch.dist(embedding1,embedding2,2).item()
			dist_list.append(dist)
			
		#store diagonal distances
		diagonal_distance=dist_list[embedding1_index]
		diag_dists.append(diagonal_distance)
		
		#calculate min distance for accuracy calcultion
		min_dist=nsmallest(1,dist_list)[0]
		min_dist_index=dist_list.index(min_dist)
		
		#print line number if requested by user 
		if args.test_run:
			if args.print_line_num:
				
				if line_counter%10==0:
				
					print('Line number:',line_counter)
		else:
			if args.print_line_num:
				
				if line_counter%10==0:
				
					print('Line number:',line_counter)
		
		#check to see if the min distance lies on the diagonal 
		#if it lies on the diagonal, the model is correct, otherwise it is wrong 
		if embedding1_index==min_dist_index:
			
			accuracy.append(1)
		else:
		
			accuracy.append(0)
		
		
		#code that stores data from 100 smallest off-diagonal distances
		#off-diag distances are all stored in the same list and may be accessed by splitting the list every 100 entries
		#for example the first 100 entries of '100_mibn_dist' contain distances from row 1 and entries 101-200 contain distances from row 2
		
		#NOTE: we add 1 to the number of k_min_dists so we can remove any instance of a diagonal distance without re-loading the data 
		k_min_dists_one_row=nsmallest(args.n_smallest_dists+1,dist_list)
		
		#write k minimum distances to the storage list 
		for off_diag_dist in k_min_dists_one_row:
			k_min_dists.append(off_diag_dist)
			
		#check to see if the diagonal distance exists in our k min distances 
		#if it does we simply remove it
		#otherwise we remove the extra distance 
		
		try:
			k_min_dists.remove(diagonal_distance)
			
			#store percent of the time we find the diagonal entry in the k min distances
			diag_found_in_k_min_dist_rate_list.append(1)
			
		except ValueError:
			end_of_row_index=args.n_smallest_dists+embedding1_index*args.n_smallest_dists
			extra_distance=k_min_dists[end_of_row_index]
			extra_distance_index=k_min_dists.index(extra_distance)
			k_min_dists.remove(extra_distance)
			diag_found_in_k_min_dist_rate_list.append(0)
			
		
		#increment index
		embedding1_index+=1
	
	print('\n')
	print('Calculations finished')
	
	#output code	
	#obtain percentage for how many times our diagonal entry is found in k min distances
	diag_found_in_k_min_dist_rate=mean(diag_found_in_k_min_dist_rate_list)
	diag_found_in_k_min_dist_rate_ouput=str('{:.0%}'.format(diag_found_in_k_min_dist_rate))
	
	print('\n')
	print('Output information:')
	print('Diagonal entry occurance in k minimum distances rate: ',diag_found_in_k_min_dist_rate_ouput)
	
	final_score=mean(accuracy)
	score_output=str('{:.0%}'.format(final_score))
	
	print('Accuracy score:')
	print(score_output)
	
	if args.print_all_data:
		
		print('Binary list for correct or incorrect:',accuracy)
		print('Diagonal distances',diag_dists)
		print('k smallest distances', k_min_dists)
		
	#saving our outputs
	
	if args.save_outputs:
		
		print('Saving Data...')
		
		data_basename=str(file_list_output[0])+'_'+str(file_list_output[1])+'_'+'data'+str(args.slice_num)+'.slice'
		
		data_save_path=args.save_path+'/'+data_basename
		
		print('File basename:',data_basename)
		print('File save path:',data_save_path)
		
		pickle.dump(accuracy,open(data_save_path+'.binary','wb'))
		pickle.dump(diag_dists,open(data_save_path+ '.diag_dists','wb'))
		pickle.dump(k_min_dists,open(data_save_path+'.k_min_dists','wb'))
		
		print('Data saving complete')
		
		
		print('Saving outputs...')
		
		output_basename=str(file_list_output[0])+'_'+str(file_list_output[1])+'_'+'summary_outputs'+str(args.slice_num)+'.slice'
		
		output_save_path=args.save_path+'/'+output_basename
		
		print('File basename:',output_basename)
		print('File save path:',output_save_path)
		
		pickle.dump(score_output,open(output_save_path+'.accuracy','wb'))
		pickle.dump(diag_found_in_k_min_dist_rate_ouput,open(output_save_path+'.diag_rate','wb'))
		pickle.dump(args.n_smallest_dists,open(output_save_path+'.value_of_k','wb'))
		
		print('Output saving complete')
		
		
accuracy_calc()




