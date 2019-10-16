# imageTestScript 
## Trainning command
--in imageTest
--python3 main.py --train -ti imageMK3/Train -si imageMK3/Test
## Image TEST
- test2.py hoge.h5 testImagePath
    - 自動トリミング + edge抽出
    - 上位3位まで表示するように変更
- h5file download
    - https://drive.google.com/open?id=11eVTIRoLKkZwPxXJqNl2bYAsHdCpTw77
        - plottype1.h5
            - label = 5 ['T', 'curve', 'cross', 'oudan', 'strate']
        - plottype2.h5
            - label.add stopLine
            - 学習画像追加
        - plottype3.h5
            - strate と stopLineのラベルの画像を一部削除
## Video TEST
- realh5.py hoge.h5 testVideoPath
    - 信頼度上位3位まで表示
    - 動画のパスを引数で取るように変更
