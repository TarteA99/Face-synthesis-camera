#必要なもののインポート
# coding: utf-8
import cv2
import numpy

#分類器のパス取得と使用可手続き
face_cascade_path = 'xmlfile/haarcascade_frontalface_default.xml'
eye_cascade_path = 'xmlfile/haarcascade_eye.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

#説明文出力
print('')
print('')
#qキーを押すとプログラム終了
print('press "q" key to shutdown')
#sキーを押すと画像保存
#保存された画像は"python/savedata/"に入れられる
print('and press "s" key to save picture')
print('')
print('Wait a minute...')

#画像の取り込みと下処理
#a
	#マスク画像をグレースケールで読み込み
a_mask_base = cv2.imread('pngfile/daimei.png',-1)
	#幅、高さ取得
a_width, a_height = a_mask_base.shape[:2]
	#アルファチャンネルのみ取り出す(ch1)
a_ch1_mask = a_mask_base[:,:,3]
	#カラー画像化(ch4)
a_color_mask = cv2.cvtColor(a_ch1_mask,cv2.COLOR_GRAY2BGR)
	#０～２５５の透明度を０～１に変換
a_color_mask = a_color_mask / 255.0
	#アルファチャンネル除去(ch3)
a_ch3_mask = a_mask_base[:,:,:3]

#b
b_mask_base = cv2.imread('pngfile/smile.png',-1)
b_width, b_height = b_mask_base.shape[:2]
b_ch1_mask = b_mask_base[:,:,3]
b_color_mask = cv2.cvtColor(b_ch1_mask,cv2.COLOR_GRAY2BGR)
b_color_mask = b_color_mask / 255.0
b_ch3_mask = b_mask_base[:,:,:3]

#c
c_mask_base = cv2.imread('pngfile/cat.png',-1)
c_width, c_height = c_mask_base.shape[:2]
c_ch1_mask = c_mask_base[:,:,3]
c_color_mask = cv2.cvtColor(c_ch1_mask,cv2.COLOR_GRAY2BGR)
c_color_mask = c_color_mask / 255.0
c_ch3_mask = c_mask_base[:,:,:3]

#d
d_mask_base = cv2.imread('pngfile/tankobu.png',-1)
d_width, d_height = d_mask_base.shape[:2]
d_ch1_mask = d_mask_base[:,:,3]
d_color_mask = cv2.cvtColor(d_ch1_mask,cv2.COLOR_GRAY2BGR)
d_color_mask = d_color_mask / 255.0
d_ch3_mask = d_mask_base[:,:,:3]

#e
e_mask_base = cv2.imread('pngfile/cat2.png',-1)
e_width, e_height = e_mask_base.shape[:2]
e_ch1_mask = e_mask_base[:,:,3]
e_color_mask = cv2.cvtColor(e_ch1_mask,cv2.COLOR_GRAY2BGR)
e_color_mask = e_color_mask / 255.0
e_ch3_mask = e_mask_base[:,:,:3]

#f
f_mask_base = cv2.imread('pngfile/mejikara.png',-1)
f_width, f_height = f_mask_base.shape[:2]
f_ch1_mask = f_mask_base[:,:,3]
f_color_mask = cv2.cvtColor(f_ch1_mask,cv2.COLOR_GRAY2BGR)
f_color_mask = f_color_mask / 255.0
f_ch3_mask = f_mask_base[:,:,:3]

#g
g_mask_base = cv2.imread('pngfile/hart.png',-1)
g_width, g_height = g_mask_base.shape[:2]
g_ch1_mask = g_mask_base[:,:,3]
g_color_mask = cv2.cvtColor(g_ch1_mask,cv2.COLOR_GRAY2BGR)
g_color_mask = g_color_mask / 255.0
g_ch3_mask = g_mask_base[:,:,:3]

#h
h_mask_base = cv2.imread('pngfile/ginbuta.png',-1)
h_width, h_height = h_mask_base.shape[:2]
h_ch1_mask = h_mask_base[:,:,3]
h_color_mask = cv2.cvtColor(h_ch1_mask,cv2.COLOR_GRAY2BGR)
h_color_mask = h_color_mask / 255.0
h_ch3_mask = h_mask_base[:,:,:3]

#i
i_mask_base = cv2.imread('pngfile/black.png',-1)
i_width, i_height = i_mask_base.shape[:2]
i_ch1_mask = i_mask_base[:,:,3]
i_color_mask = cv2.cvtColor(i_ch1_mask,cv2.COLOR_GRAY2BGR)
i_color_mask = i_color_mask / 255.0
i_ch3_mask = i_mask_base[:,:,:3]

#j
j_mask_base = cv2.imread('pngfile/lab.png',-1)
j_width, j_height = j_mask_base.shape[:2]
j_ch1_mask = j_mask_base[:,:,3]
j_color_mask = cv2.cvtColor(j_ch1_mask,cv2.COLOR_GRAY2BGR)
j_color_mask = j_color_mask / 255.0
j_ch3_mask = j_mask_base[:,:,:3]

#k
k_mask_base = cv2.imread('pngfile/miserarenai.png',-1)
k_width, k_height = k_mask_base.shape[:2]
k_ch1_mask = k_mask_base[:,:,3]
k_color_mask = cv2.cvtColor(k_ch1_mask,cv2.COLOR_GRAY2BGR)
k_color_mask = k_color_mask / 255.0
k_ch3_mask = k_mask_base[:,:,:3]

#l
l_mask_base = cv2.imread('pngfile/hamaki.png',-1)
l_width, l_height = l_mask_base.shape[:2]
l_ch1_mask = l_mask_base[:,:,3]
l_color_mask = cv2.cvtColor(l_ch1_mask,cv2.COLOR_GRAY2BGR)
l_color_mask = l_color_mask / 255.0
l_ch3_mask = l_mask_base[:,:,:3]

#m
m_mask_base = cv2.imread('pngfile/ika.png',-1)
m_width, m_height = m_mask_base.shape[:2]
m_ch1_mask = m_mask_base[:,:,3]
m_color_mask = cv2.cvtColor(m_ch1_mask,cv2.COLOR_GRAY2BGR)
m_color_mask = m_color_mask / 255.0
m_ch3_mask = m_mask_base[:,:,:3]

#n
n_mask_base = cv2.imread('pngfile/hamaki2.png',-1)
n_width, n_height = n_mask_base.shape[:2]
n_ch1_mask = n_mask_base[:,:,3]
n_color_mask = cv2.cvtColor(n_ch1_mask,cv2.COLOR_GRAY2BGR)
n_color_mask = n_color_mask / 255.0
n_ch3_mask = n_mask_base[:,:,:3]

#o
o_mask_base = cv2.imread('pngfile/eye.png',-1)
o_width, o_height = o_mask_base.shape[:2]
o_ch1_mask = o_mask_base[:,:,3]
o_color_mask = cv2.cvtColor(o_ch1_mask,cv2.COLOR_GRAY2BGR)
o_color_mask = o_color_mask / 255.0
o_ch3_mask = o_mask_base[:,:,:3]

#p
p_mask_base = cv2.imread('pngfile/fukidasikoukoku.png',-1)
p_width, p_height = p_mask_base.shape[:2]
p_ch1_mask = p_mask_base[:,:,3]
p_color_mask = cv2.cvtColor(p_ch1_mask,cv2.COLOR_GRAY2BGR)
p_color_mask = p_color_mask / 255.0
p_ch3_mask = p_mask_base[:,:,:3]

#r

#s


#t
t_mask_base = cv2.imread('pngfile/hige.png',-1)
t_width, t_height = t_mask_base.shape[:2]
t_ch1_mask = t_mask_base[:,:,3]
t_color_mask = cv2.cvtColor(t_ch1_mask,cv2.COLOR_GRAY2BGR)
t_color_mask = t_color_mask / 255.0
t_ch3_mask = t_mask_base[:,:,:3]

#u
u_mask_base = cv2.imread('pngfile/money.png',-1)
u_width, u_height = u_mask_base.shape[:2]
u_ch1_mask = u_mask_base[:,:,3]
u_color_mask = cv2.cvtColor(u_ch1_mask,cv2.COLOR_GRAY2BGR)
u_color_mask = u_color_mask / 255.0
u_ch3_mask = u_mask_base[:,:,:3]

#v
v_mask_base = cv2.imread('pngfile/fukidasi.png',-1)
v_width, v_height = v_mask_base.shape[:2]
v_ch1_mask = v_mask_base[:,:,3]
v_color_mask = cv2.cvtColor(v_ch1_mask,cv2.COLOR_GRAY2BGR)
v_color_mask = v_color_mask / 255.0
v_ch3_mask = v_mask_base[:,:,:3]

#w
w_mask_base = cv2.imread('pngfile/bike.png',-1)
w_width, w_height = w_mask_base.shape[:2]
w_ch1_mask = w_mask_base[:,:,3]
w_color_mask = cv2.cvtColor(w_ch1_mask,cv2.COLOR_GRAY2BGR)
w_color_mask = w_color_mask / 255.0
w_ch3_mask = w_mask_base[:,:,:3]

#x
x_mask_base = cv2.imread('pngfile/pokego.png',-1)
x_width, x_height = x_mask_base.shape[:2]
x_ch1_mask = x_mask_base[:,:,3]
x_color_mask = cv2.cvtColor(x_ch1_mask,cv2.COLOR_GRAY2BGR)
x_color_mask = x_color_mask / 255.0
x_ch3_mask = x_mask_base[:,:,:3]

#y
y_mask_base = cv2.imread('pngfile/poke.png',-1)
y_width, y_height = y_mask_base.shape[:2]
y_ch1_mask = y_mask_base[:,:,3]
y_color_mask = cv2.cvtColor(y_ch1_mask,cv2.COLOR_GRAY2BGR)
y_color_mask = y_color_mask / 255.0
y_ch3_mask = y_mask_base[:,:,:3]

#z
z_mask_base = cv2.imread('pngfile/insta.png',-1)
z_width, z_height = z_mask_base.shape[:2]
z_ch1_mask = z_mask_base[:,:,3]
z_color_mask = cv2.cvtColor(z_ch1_mask,cv2.COLOR_GRAY2BGR)
z_color_mask = z_color_mask / 255.0
z_ch3_mask = z_mask_base[:,:,:3]

#ビデオ使用可手続き
cap = cv2.VideoCapture(0)

#初期値設定
fin = ord('a')

#メインループ
while True:
	frag = 0
#キー入力の継続手続き
	key = cv2.waitKey(50)&0xff
	if key == ord('a'):
		fin = ord('a')
	elif key == ord('b'):
		fin = ord('b')
	elif key == ord('c'):
		fin = ord('c')
	elif key == ord('d'):
		fin = ord('d')
	elif key == ord('e'):
		fin = ord('e')
	elif key == ord('f'):
		fin = ord('f')
	elif key == ord('g'):
		fin = ord('g')
	elif key == ord('h'):
		fin = ord('h')
	elif key == ord('i'):
		fin = ord('i')
	elif key == ord('j'):
		fin = ord('j')
	elif key == ord('k'):
		fin = ord('k')
	elif key == ord('l'):
		fin = ord('l')
	elif key == ord('m'):
		fin = ord('m')
	elif key == ord('n'):
		fin = ord('n')
	elif key == ord('o'):
		fin = ord('o')
	elif key == ord('p'):
		fin = ord('p')
	elif key == ord('r'):
		fin = ord('r')
	elif key == ord('s'):
#画像の保存
		cv2.imwrite('savedata/kobe-kosen-densi-yuki.jpg',img)
	elif key == ord('t'):
		fin = ord('t')
	elif key == ord('u'):
		fin = ord('u')
	elif key == ord('v'):
		fin = ord('v')
	elif key == ord('w'):
		fin = ord('w')
	elif key == ord('x'):
		fin = ord('x')
	elif key == ord('y'):
		fin = ord('y')
	elif key == ord('z'):
		fin = ord('z')

#キー入力によって読み込みファイル変更
	if fin == ord('a'):
		frag = 3
		color_mask = a_color_mask
		ch3_mask = a_ch3_mask
	elif fin == ord('b'):
		color_mask = b_color_mask
		ch3_mask = b_ch3_mask
	elif fin == ord('c'):
		color_mask = c_color_mask
		ch3_mask = c_ch3_mask
	elif fin == ord('d'):
		color_mask = d_color_mask
		ch3_mask = d_ch3_mask
	elif fin == ord('e'):
		color_mask = e_color_mask
		ch3_mask = e_ch3_mask
	elif fin == ord('f'):
		color_mask = f_color_mask
		ch3_mask = f_ch3_mask
	elif fin == ord('g'):
		frag = 2
		color_mask = g_color_mask
		ch3_mask = g_ch3_mask
	elif fin == ord('h'):
		frag = 3
		color_mask = h_color_mask
		ch3_mask = h_ch3_mask
	elif fin == ord('i'):
		color_mask = i_color_mask
		ch3_mask = i_ch3_mask
	elif fin == ord('j'):
		color_mask = j_color_mask
		ch3_mask = j_ch3_mask
	elif fin == ord('k'):
		color_mask = k_color_mask
		ch3_mask = k_ch3_mask
	elif fin == ord('l'):
		color_mask = l_color_mask
		ch3_mask = l_ch3_mask
	elif fin == ord('m'):
		frag = 3
		color_mask = m_color_mask
		ch3_mask = m_ch3_mask
	elif fin == ord('n'):
		color_mask = n_color_mask
		ch3_mask = n_ch3_mask
	elif fin == ord('o'):
		color_mask = o_color_mask
		ch3_mask = o_ch3_mask
	elif fin == ord('p'):
		frag = 3
		color_mask = p_color_mask
		ch3_mask = p_ch3_mask
	elif fin == ord('r'):
		frag = 1
#	elif fin == ord('s'):
	elif fin == ord('t'):
		color_mask = t_color_mask
		ch3_mask = t_ch3_mask
	elif fin == ord('u'):
		frag = 2
		color_mask = u_color_mask
		ch3_mask = u_ch3_mask
	elif fin == ord('v'):
		frag = 3
		color_mask = v_color_mask
		ch3_mask = v_ch3_mask
	elif fin == ord('w'):
		frag = 3
		color_mask = w_color_mask
		ch3_mask = w_ch3_mask
		
	elif fin == ord('x'):
		frag = 3
		color_mask = x_color_mask
		ch3_mask = x_ch3_mask
	elif fin == ord('y'):
		frag = 3
		color_mask = y_color_mask
		ch3_mask = y_ch3_mask
	elif fin == ord('z'):
		frag = 3
		color_mask = z_color_mask
		ch3_mask = z_ch3_mask

#カメラ画像の取得
	ret, img     = cap.read()

#グレイスケール変換
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#顔認識
	results = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=3, minSize=(100, 100))

#モザイク処理
	if frag == 1:
		for x,y,w,h in results:
	#画像の切り抜き
			crop_img = img[y:y+h,x:x+w]
	#ピクセル数の小さな画像にリサイズ
			crop_img = cv2.resize(crop_img,(15,15))
	#元のピクセルに再変換
			crop_img = cv2.resize(crop_img,(h,w))
	#表示準備
			img[y:y+h, x:x+w] = crop_img

#瞳認証処理と画像合成
	elif frag == 2:
		for x,y,w,h in results:
	#グレースケール変換
			face_gray = img_gray[y: y + h, x: x + w]
	#瞳認識
			eresults = eye_cascade.detectMultiScale(face_gray, scaleFactor=2.3, minNeighbors=3)
			for yx,yy,ew,eh in eresults:
	#変数計算
				ex = yx + x
				ey = yy + y
	#画像の切り抜き
				crop_img = img[ey:ey+eh,ex:ex+ew]
	#瞳認識処理
				cv2.rectangle(crop_img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
	#アルファチャンネルマスクのリサイズ
				re_color_mask = cv2.resize(color_mask,(ew,eh))
	#マスクのリサイズ
				re_ch3_mask = cv2.resize(ch3_mask,(ew,eh))
	#画像合成準備
				img_sum = cv2.resize(crop_img,(ew,eh))
	#和、差、積を使用し画像合成し表示準備
				img[ey:ey+eh, ex:ex+ew] = img_sum*(1-re_color_mask)
				crop_img = img[ey:ey+eh,ex:ex+ew]
				img_last = cv2.resize(crop_img,(ew,eh))
				img[ey:ey+eh, ex:ex+ew] = img_last + re_ch3_mask *re_color_mask

#背景合成処理
	elif frag == 3:
		for x,y,w,h in results:
	#画像の幅、高さ取得
			i_w,i_h = img.shape[:2]
	#画像の切り抜き
			crop_img = cv2.resize(img,(i_h,i_w))
	#アルファチャンネルマスクのリサイズ
			re_color_mask = cv2.resize(color_mask,(i_h,i_w))
	#マスクのリサイズ
			re_ch3_mask = cv2.resize(ch3_mask,(i_h,i_w))
	#画像合成準備
			img_sum = cv2.resize(crop_img,(i_h,i_w))
	#和、差、積を使用し画像合成し表示準備
			img[0:i_w, 0:i_h] = img_sum*(1-re_color_mask)
			crop_img = cv2.resize(img,(i_h,i_w))
			img_last = cv2.resize(crop_img,(i_h,i_w))
			img[0:i_w,0:i_h] = img_last + re_ch3_mask *re_color_mask

#顔合成処理
	elif frag == 0:
		for x,y,w,h in results:
	#画像の切り抜き
			crop_img = img[y:y+h,x:x+w]
	#アルファチャンネルマスクのリサイズ
			re_color_mask = cv2.resize(color_mask,(w,h))
	#マスクのリサイズ
			re_ch3_mask = cv2.resize(ch3_mask,(w,h))
	#画像合成準備
			img_sum = cv2.resize(crop_img,(w,h))
	#和、差、積を使用し画像合成し表示準備
			img[y:y+h, x:x+w] = img_sum*(1-re_color_mask)
			crop_img = img[y:y+h,x:x+w]
			img_last = cv2.resize(crop_img,(w,h))
			img[y:y+h, x:x+w] = img_last + re_ch3_mask *re_color_mask

# 表示
	#画像名設定
	cv2.namedWindow('yuki',cv2.WINDOW_NORMAL)
	#フルスクリーン設定
	cv2.setWindowProperty('yuki',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
	#画像表示
	cv2.imshow('yuki', img)
	#終了処理
	if key == ord('q'):
		break
#カメラ終了
cap.release()
#全画面close
cv2.destroyAllWindows()
