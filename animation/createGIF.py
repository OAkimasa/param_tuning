import glob
from PIL import Image
 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.core.fromnumeric import sort

#フォルダ名を入れます
folderName = "Last_Macro_Anime2"

#該当フォルダから画像のリストを取得。読み込みたいファイル形式を指定。ここではpng
#picList = glob.glob(folderName + "\*.png")  # windows
picList = glob.glob(folderName + "/*.png")  # mac
picList = sort(picList)
print(picList)

#figオブジェクトの作成
fig = plt.figure(figsize=(18,18))

#figオブジェクトから目盛り線などを消す
plt.axis('off')

#空のリスト作成
ims = []

#画像ファイルを空のリストの中に1枚ずつ読み込み
for i in range(len(picList)):
    #読み込んで付け加えていく
    tmp = Image.open(picList[i])
    ims.append([plt.imshow(tmp)])

#アニメーション作成
ani = animation.ArtistAnimation(fig, ims, blit=True)

#アニメーション保存。ファイル名を入力
ani.save("Last_Macro_Anime2.gif", writer='pillow')
