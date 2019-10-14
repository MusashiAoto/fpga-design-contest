import sys,os,cv2,shutil

savePath = "trimmingImg"

if os.path.isdir(savePath):
	shutil.rmtree(savePath)
os.makedirs(savePath,exist_ok=True)


def trimming(path):
	files = [os.path.abspath(os.path.join(path,x)) for x in os.listdir(path)]
	for f in files:
		if os.path.isdir(f): trimming(f)
		if os.path.splitext(f)[1] in [".bmp",".png",".jpg",".jpeg"]:
			img = readAndTrimming(f)

			mkname = os.path.join(savePath,os.path.dirname(f).split("/")[-1])
			if not os.path.isdir(mkname): os.makedirs(mkname)
			cv2.imwrite(os.path.join(mkname,os.path.basename(f)),img)
			# print(os.path.join(savePath,os.path.dirname(f),os.path.basename(f)))
			# cv2.imwrite(os.path.join(savePath,))

def readAndTrimming(path):
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    size = img.shape
    img = img[:int(size[0]/4),:]

    # img = cv2.Canny(img, 105, 110)
    img = cv2.Canny(img, 15, 106)
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    # img = cv2.resize(img, (Width, Height))
    return img

if len(sys.argv) != 2:
	print("error")
	sys.exit(1)

trimming(sys.argv[1])
