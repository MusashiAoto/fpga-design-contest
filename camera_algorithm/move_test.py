import cv2
import numpy as np
import time



fractal_controll=0
minLineLength = 1000
maxLineGap = 1
lines=0
defcount=0
Xpoint=145
avecontrast=0
contraststate=0
fps=30
Height=850
Width=500
magaru= cv2.imread("image/curve.png" , 0)
oudan = cv2.imread("image/oudan.png" , 0)
rT= cv2.imread("sozai/curve4.png" ,0)

front_time=time.time()
def detect(ip,detect_source):

    # 対象画像を指定


    # 画像をグレースケールで読み込み
    gray_base_src = ip
    gray_temp_src = detect_source

    # マッチング結果書き出し準備
    # 画像をBGRカラーで読み込み
    #color_base_src = cv2.imread(base_image_path, 1)
    #color_temp_src = cv2.imread(temp_image_path, 1)

    # 特徴点の検出
    type = cv2.AKAZE_create()
    kp_01, des_01 = type.detectAndCompute(gray_base_src, None)
    kp_02, des_02 = type.detectAndCompute(gray_temp_src, None)

    # マッチング処理
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    try:
        matches = bf.match(des_01, des_02)
        matches = sorted(matches, key = lambda x:x.distance)
        #mutch_image_src = cv2.drawMatches(color_base_src, kp_01, color_temp_src, kp_02, matches[:10], None, flags=2)


        dist = [m.distance for m in matches]
        ret = sum(dist) / len(dist)
        #print(ret)
        return ret
    except :
        print("E")
        return 0



def chokan(img):
    img1=img
    height, width, channels = img.shape[:3]
    hidariueX=int(width/320*100)-20
    hidariueY=int(height/240*40)
    migiueX=int(width/320*210)+20
    migiueY=int(height/240*40)


    #image = np.zeros((700, 700, 3), np.uint8)
    #src = np.array([[100,40],[210,40],[0,239],[width,239]],np.float32)
    src = np.array([[hidariueX,hidariueY],[migiueX,migiueY],[0,height],[width,height]],np.float32)

    dst = np.array([[0,0],[Width,0],[int(Width/5),Height],[int(Width/5*4),Height]],np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warp = cv2.warpPerspective(img1.copy(), M, (Width, Height))

    return warp
    #cv2.imshow('transform', warp)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.imwrite("convert.png",warp)




cap = cv2.VideoCapture("output5.m4v")
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') 
convert_out = cv2.VideoWriter('outpasdaderut2.m4v',fourcc, fps, (Width,Height))

before=185
after=0

#hosei you
# x47 y 641
# x82
# x23
def curvedetect(out):
    pxL = out[int(Height/4*3),int(Width/4)]
    pxR = out[int(Height/2),int(Width/4*3)]
    pxC = out[226,488]
    #print(pxL)
    #print(pxR)
    #out=cv2.circle(out, (int(Width/4*3),int(Height/2)), 10, color=(0, 255, 0), thickness=-1)
    #out=cv2.circle(out, (int(Width/4),int(Height/4*3)), 10, color=(255, 0, 0), thickness=-1)
    if pxR==pxL==255 :
        print("curve")
        #out=cv2.putText(out, 'Curve', (300, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)


        return 1



def curveV(out):

    out1 = out[int(Height/3):int(Height/4*3),0:Width]
    c=0
    Vc=0
    for x in range(int(Height/4*3)-int(Height/3)):
        PC=out1[x,int(Width/4)]

        if PC==255:
            c=x
            break
    for x in range(int(Height/4*3)-int(Height/3)):
        PV=out1[x,int(Width/4*3)]
        if PV==255:
            Vc=x
            break
    print(c,Vc)
    if c-Vc<150 and 90<c-Vc:
        print("curve")
        #cv2.imwrite("detect.png",out1)
        #out=cv2.putText(out, 'Curve', (300, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)


        #return 1

    return out1

    # pxL = out[int(Height/8*7),int(Width/4)]
    # pxR = out[int(Height*0.625)+15,int(Width/4*3)]
    # pxC = out[226,488]
    # #print(pxL)
    # #print(pxR)
    # out=cv2.circle(out, (int(Width/4*3),int(Height*0.625)+15), 10, color=(0, 255, 0), thickness=-1)
    # out=cv2.circle(out, (int(Width/4),int(Height/8*7)), 10, color=(255, 0, 0), thickness=-1)
    # if pxR==pxL==255 :
    #     print("curve")
    #     #out=cv2.putText(out, 'Curve', (300, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)


    #     return 1


def dete(source,base):

    
    #画像をグレースケールで読み込む
    #img = cv2.imread(source, 0)
    #temp = cv2.imread(base, 0)
    bairitu=0.4
    img = cv2.resize(source, dsize=None, fx=bairitu, fy=bairitu)
    temp=cv2.resize(base, dsize=None, fx=bairitu, fy=bairitu)

    #マッチングテンプレートを実行
    try:
        result = cv2.matchTemplate(img, temp, cv2.TM_CCOEFF_NORMED)
        #類似度の設定(0~1)
        threshold = 0.9
        #検出結果から検出領域の位置を取得
        loc = np.where(result >= threshold)
        #検出領域を四角で囲んで保存
        result = img
        w, h = temp.shape[::-1]
        for top_left in zip(*loc[::-1]):
            bottom_right = (top_left[0] + w, top_left[1] + h)
        
        cv2.rectangle(result,top_left, bottom_right, (255, 0, 0), 2)
        cv2.imwrite("result2ss.png", result)


        cv2.imshow("mutch_image_src", result)
        #cv2.imshow("02_result08", mutch_image_src)

        #cv2.waitKey(1)
        print("detect")
        return result
    except :
       
        return source

def stopdetect(out):
    out = out[int(Height/2):int(Height/2+20),int(Width/4):int(Width/4*3)]

    
    #print(int(Height/2),int(Width/4),int(Height/2+20),int(Width/4*3))
    #cv2.imwrite("aaa.png",out)
    #cv2.imshow("box", out)
    #cv2.waitKey(1)
    binary = cv2.inRange(out, 254, 255)

    # 画素が1の画素数を数える。
    cnt = cv2.countNonZero(binary)

    if cnt>4000:
        print("stop")
        return 1

def Tdetect(out):
    g=250
    h=30
    out1 = out[231+g:390+g,374+h:410+h]
    out2 = out[298+g:333+g,374+h:500+h]
    
    #print(int(Height/2),int(Width/4),int(Height/2+20),int(Width/4*3))
    #cv2.imwrite("aaa.png",out)
    cv2.imshow("box1", out1)
    cv2.imshow("box2", out2)
    cv2.waitKey(1)
    binary = cv2.inRange(out1, 254, 255)
    binary2 = cv2.inRange(out2, 254, 255)
    # 画素が1の画素数を数える。
    cnt1 = cv2.countNonZero(binary)
    cnt2 = cv2.countNonZero(binary2)

    print(cnt1,cnt2)
    if cnt1>3200 and cnt2>1900:
        print("TT")
        return 1



def nomaldetect(out):

    pxL = out[int(Height/8*7),int(Width/4)+5]#kokodechosei

    if pxL==255:
        print("chosei")

frame_cout=0
target=[]
def readmap(map):
    

    f = open(map)
    data1 = f.read() 
    f.close()

    lines1 = data1.split('\n')

    for i in lines1:

        target.append(i.split(','))
    

map = "yosen_map.txt"
readmap(map)


tar_cnt=0
navi,angleL,angleR=target[tar_cnt]
i=0
while(cap.isOpened()):
    frame_cout+=1
    if frame_cout==1:
        
        print("START")



    #img1 = cv2.imread("frame.png", cv2.IMREAD_COLOR)
    # cv2.imshow('transform', img1)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ret,img1= cap.read()
    if ret==False:
        break
    out=chokan(img1)
    imm= out
    out  = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    ret2, otsu = cv2.threshold(out, 0, 255, cv2.THRESH_OTSU)
    if abs(ret2-before)>30:
        ret,  out= cv2.threshold(out, before, 255, cv2.THRESH_BINARY)
    else:
        before=ret2
        out=otsu
    #print(ret2)
    

    # rt1=detect(out,magaru)
    # rt2=detect(out,oudan)
    # rt3=detect(out,rT)

    # if rt1>rt2 and rt1>rt3:
    #     print("curve")
    # elif rt2>rt1 and rt2>rt3:
    #     print("oudan")
    # elif rt3>rt2 and rt3>rt1:
    #     print("T")

    #----------------Detectors--------------


    # if navi=="r":
    #     D=curvedetect(out)
    #     if D==1:
    #         #print(angle)
    #         tar_cnt=tar_cnt+1
    #         navi,angleL,angleR=target[tar_cnt]
    #         print("next:"+navi)
    #         #print(target)
            
    # elif navi=="t":
    #     #detectT
    #     if D==Tdetect(out):
    #         tar_cnt=tar_cnt+1
    #         navi,angleL,angleR=target[tar_cnt]
    #         print("next:"+navi)
    # elif navi=="-":
    #     D=stopdetect(out)
    #     if D==1:
    #         #print(angle)
    #         tar_cnt+=1
    #         navi,angleL,angleR=target[tar_cnt]
    #         print("next:"+navi)


    #out=curveV(out)
    #stopdetect(out)
    #nomaldetect(out)
    out=dete(out,rT)


    out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    # out = cv2.line(out,(int(Width/4),int(Height/2)),(int(Width/4),Height),(255,0, 0),5)
    # out = cv2.line(out,(0,int(Height/4*3)),(Width,int(Height/4*3)),(255,0, 0),5)




    
    # out = cv2.line(out,(int(Width/4*3),0),(int(Width/4*3),Height),(0,255, 0),5)
    # out = cv2.line(out,(0,int(Height/2)),(Width,int(Height/2)),(0,255, 0),5)

    #cv2.imshow("window_name", out)
    #name="movies/"+str(frame_cout)+".png"
    #cv2.imwrite(name,out)
    #time.sleep(0.05)
    convert_out.write(out)
    #cv2.imwrite("img/"+ str(i)+".png",out)

    #i+=1


    #cv2.waitKey(0)
convert_out.release()
cv2.destroyAllWindows()