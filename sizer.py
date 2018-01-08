import os
import Image
import sys
output=None
def lser(folderNum):
	lsPipe=os.popen('ls %d/'%folderNum)
	while True:
		fileName=lsPipe.readline()
		if not fileName:
			break
		if sys.argv[2]=='p':
			#print(fileName[0:len(fileName)-1])
			im=Image.open('%d/%s'%(folderNum, fileName[0:len(fileName)-1]))
			x,y=im.size
			output.writelines('%d/%s 1 0 0 %d %d\n'%(folderNum, fileName[0:len(fileName)-1], x, y))
		else:
			output.writelines('%s/%d/%s\n'%(sys.argv[3],folderNum, fileName[0:len(fileName)-1]))
	

def wrongUsage():
	print("wrong usage!!")
	print("sizer.py [num of folder to use] [p or n] [parent folder]")
	sys.exit(1)

if __name__ == "__main__":
	if len(sys.argv)<3:
		wrongUsage()
	if sys.argv[2]!='p' and sys.argv[2]!='n':
		wrongUsage()
	output=open('imagelist.txt','w')
	for i in range(int(sys.argv[1])):
		lser(i)
	output.close()




