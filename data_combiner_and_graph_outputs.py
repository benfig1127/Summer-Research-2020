import argparse
import os
import glob
import pickle
parser = argparse.ArgumentParser(fromfile_prefix_chars='@')


parser_data = parser.add_argument_group('data options')
parser_data.add_argument('--data_loc',type=str)
parser_data.add_argument('--data_endings','--list', nargs='+',help='<Required> Set flag', required=True)
parser_data.add_argument('--save_path',type=str)
parser_data.add_argument('--lang',type=str)


args = parser.parse_args()


data_endings_list=args.data_endings
print('\n')
print('Data filename endings:',data_endings_list)
print('\n')

for ending in data_endings_list:
	filename_list=glob.glob(args.data_loc+'/*'+ending)
	short_filename_list=[]
	for long_filename in filename_list:
		short_filename_list.append(os.path.basename(long_filename))
	
	print('Files to be combined:',short_filename_list)
	
	combined_file=[]
	for filename in filename_list:
			with open(filename,'rb') as pickled_file:
				file=pickle.load(pickled_file)
			try:
				print('First 10 items',file[:10],'\n')
			except:
				print('Items:',file,'\n')
			try:
				combined_file.extend(file)
			except:
				combined_file.append(file)
	pickle.dump(combined_file,open(args.save_path+'/'+args.lang+str(ending)+'.combined','wb'))
