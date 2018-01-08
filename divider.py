import os

def moveFiles(file_stack, folder_index):
	os.system('mkdir %d'%folder_index)
	for filename in file_stack:
		os.system('mv %s %d'%(filename[0:len(filename)-1], folder_index))

if __name__ == "__main__":
	file_object = open('filelist.txt')
	file_stack=[]
	folder_index=0
	while True:
		line=file_object.readline()
		if not line:
			break
		file_stack.append(line)
		if len(file_stack)%100==0:
			moveFiles(file_stack, folder_index)
			folder_index+=1
			file_stack=[]
		#print(file_stack)
	if len(file_stack)>0:
		moveFiles(file_stack, folder_index)



