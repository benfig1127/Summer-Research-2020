import argparse
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')


parser_data = parser.add_argument_group('data options')
parser_data.add_argument('--data',type=str)
#parser_data.add_argument('--langs','--list', nargs='+',help='<Required> Set flag', required=True)

parser_print=parser.add_argument_group('print options')
parser_print.add_argument('--file_names',action='store_true')


args = parser.parse_args()

import torch
import os
import glob
from statistics import mean
import pandas as pd

if args.data is None: 
	raise ValueError('Please specify data folder')
#file1_name=glob.glob(args.data+'/*'+args.langs[0]+'.bert')[0]
#file2_name=glob.glob(args.data+'/*'+args.langs[1]+'.bert')[0]

#file1=torch.load(file1_name)
#file2=torch.load(file2_name)




file_list1=glob.glob(args.data+'/*'+'.bert')
file_list_output1=[]



for file in file_list1:
	filename=os.path.basename(file)
	short_filename=filename.split('6way.')[1].lstrip().split('.')[0]
	file_list_output1.append(short_filename)

file_list2=file_list1
file_list_output2=file_list_output1
df=pd.DataFrame(index=file_list_output1,columns=file_list_output2)

print(df)

if args.file_names:
	print(file_list_output1)


def accuracy_calc():
	for j in file_list2:
		for i in file_list1:
			file1=torch.load(i)
			file2=torch.load(j)
			accuracy=[]
			embedding1_index=1
			for embedding1 in file1:	
				dist_list=[]
				for embedding2 in file2: 
					dist=round(torch.dist(embedding1,embedding2,2).item(),4)
					dist_list.append(dist)
				min_dist=min(dist_list)
				min_dist_index=dist_list.index(min_dist)+1
				

				if embedding1_index==min_dist_index:
					
					accuracy.append(1)
				else:
				
					accuracy.append(0)
				
				embedding1_index+=1
				final_score=mean(accuracy)
				score_output=str('{:.0%}'.format(final_score))
				#print(score_output)
			df.iloc[file_list1.index(i)][file_list2.index(j)]=score_output
			print(df)
		print(df)
	print(df)
	print('Rows indicate the input language and the columns indicate the language the input is compared to') 
	df.to_csv('accuracy_outputs.csv',index=False)
accuracy_calc()




