import os,platform,re,sys
from datetime import datetime
# 入力サイズ
Height = int(320/4)
Width = int(480/4)

# 入力変数のサイズ
# 変数の入力サイズを使用する場合、Variable_input以下はTrueです。
# 画像は、長辺（幅または高さ）がMax_sideに等しいことを満たしてサイズ変更されます。
# Variable_inputがTrueの場合、 "Height"と "Width"の上は無視されます。
Variable_input =  False
Max_side =  1024

Test_Max_side =  1024  # 1536

# 入力データの形状
# 1 channels_last - > [MB、C、H、W]、channels_first - > [MB、H、W、C]
Input_type = 'channels_last'

File_extensions = ['.jpg', '.png','.bmp']


## Training config
Step = 100
SvStep = 2
Minibatch = 32
Learning_rate = 0.001

Test_Minibatch = 32

## Data augmentation
Horizontal_flip = False
Vertical_flip = False
Rotate_ccw90 = False
bottom_trimming = False

## Save config
Save_dir = 'out'
Ir_dir = "ir2"
time = datetime.now().strftime("%Y%m%d%H%M%S")
Model_name = str(time)+'_e'+str(Step)+'_CNN.h5'
Save_path = os.path.join(Save_dir, Model_name)
os.makedirs(os.path.join(Save_dir,Ir_dir),exist_ok=True)
os.makedirs(Save_dir,exist_ok=True)

Random_seed = 0

Load_Model_List = [x for x in os.listdir(os.path.join(os.getcwd(),Save_dir)) if x in [".npz"]]

if len(Load_Model_List)>0:
	Load_Model_Path = os.path.join(Save_dir, os.listdir(os.getcwd()+"/"+Save_dir)[-1])
else:
	Load_Model_Path=""

# メモ作り
Memo_dir = "memo"
Memo_name = str(time)+'_e'+str(Step)+'_CNN.txt'
Memo_Path = os.path.join(Memo_dir, Memo_name)

Class_Path = ""
Test_Path = ""

# 学習画像のパス (macとubuntu用)
# if platform.system() == "Darwin":
# 	Class_Path = "/Users/syoya/GoogleDriveUniv/lab/uocr/font2img/img"
# 	Test_Path = "/Users/syoya/GoogleDriveUniv/lab/uocr/font2img/tst"
# elif platform.system() == "Linux":
# 	Class_Path = "/media/syoya/HDD/uocrKeras/font2img/zzz_3c2f3p/Train"
# 	Test_Path = "/media/syoya/HDD/uocrKeras/font2img/zzz_3c2f3p/Test"
# else:
# 	print("error : cnnConfig.py ONLY MAC OR UBUNTU.")
# 	exit(1)

# ディレクトリの名前とラベル用の名前
ngNameChr = ["\\", "/", ":", "*", "?", "\"", "<", ">", "|", ",", ".","-","0","1","2","3","4","5","6","7","8","9"]
replaceChr = ["bs","sl","cl","ar","qm","dc","sn","dn","vb","cm","pr","hf","zr","ow","tw","th","fr","fv","sx","sv","et","nn"]


# 学習画像のパスが存在しないときのエラー
# if not os.path.lexists(Class_Path):
# 	print("error : cnnConfig.py \"Class_Path\" No such directory")
# 	exit(1)

# クラスラベル
Class_label = []
Class_num = 0

# for i, c in enumerate(Class_label):
# 	if c in replaceChr: Class_label[i] = ngNameChr[replaceChr.index(c)]

# クラス数
# Class_num = len(Class_label)

# 学習データのパス
# Train_dirs = sorted([Class_Path+"/"+x for x in os.listdir(Class_Path) if os.path.isdir(Class_Path+"/"+x)])
Train_dirs = []
# テストデータのパス
# Test_dirs = sorted([Test_Path+"/"+x for x in os.listdir(Test_Path) if os.path.isdir(Test_Path+"/"+x)])
Test_dirs = []

# テストデータ数
Test_num = 0
# for f in Test_dirs:
# 	Test_num += len(os.listdir(f))


def setImage(cp, tp):
	if cp == None:
		print("error config.py setImage() cp == None")
		sys.exit(1)

	global Class_Path,Class_label,Class_num,Train_dirs,Test_Path,Test_dirs

	Class_Path = cp
	Class_label = [x for x in os.listdir(Class_Path) if os.path.isdir(Class_Path+"/"+x)]
	Class_num = len(Class_label)

	Train_dirs = sorted([Class_Path+"/"+x for x in os.listdir(Class_Path) if os.path.isdir(Class_Path+"/"+x)])

	if tp != None:
		Test_Path = tp
		Test_dirs = sorted([Test_Path+"/"+x for x in os.listdir(Test_Path) if os.path.isdir(Test_Path+"/"+x)])

# 出力用
print("time:",time)
print("pwd:",os.getcwd())
print("Height x Width: {} x {}".format(Height,Width))
print("epoch:",Step)
print("saveEpoch:",SvStep)
print()
print("Test_Path",Test_Path)
print("Test_dirs",Test_dirs)




if __name__ == "__main__":
	print("epoch :",Step)
	print("len(Load_Model_List):",len(Load_Model_List))
