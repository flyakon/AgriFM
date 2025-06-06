import os
def get_filename(file_path,is_suffix=True)->str:
	file_name=file_path.replace('/','\\')
	file_name=file_name.split('\\')[-1]
	if is_suffix:
		return file_name
	else:
		index=file_name.rfind('.')
		if index>0:
			return file_name[0:index]
		else:
			return file_name

def get_parent_folder(file_path,with_root=False):

	file_path=file_path.replace('\\','/')


	index = file_path.rfind('/')
	parent_folder=file_path[0:index]
	if not with_root:
		return get_filename(parent_folder)
	return parent_folder


def split_filename(file_path:str,split_str:str)->(str,str):
	'''
	根据split_str将文件分为两部分，split_str为后半部分
	:param file_path:
	:param split_str:
	:return:
	'''
	index=file_path.index(split_str)
	return file_path[0:index],file_path[index:]

def get_root_path(data_file):
	file_path = data_file.replace('\\', '/')

	index = file_path.find('/')
	parent_folder = file_path[0:index]
	return parent_folder

if __name__=='__main__':
	file_name='GF1_WFV1_E80.0_N29.6_20200920_L1A0005075711_GEO_4488_816.tif'
	print(split_filename(file_name,'_GEO'))