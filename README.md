Graded Response Model のパラメータ推定
  
# usage  
```commandline  
$ python main.py response_csv
```
response_csvには以下のようなcsvファイルのパスを指定。
| カラム名 | 説明 |  
|-------:|:-----|  
|item|プレイヤの識別子 <br>(プレイヤごとに異なる何らかの文字列)|  
|person|被検者の識別子 <br>(被検者ごとに異なる何らかの文字列)|  
|response|被検者の項目に対する反応 <br>(1以上の整数)|

```
(例)
item,person,response
i1,p1,1
i1,p2,3
i2,p1,2
i3,p1,2
...
```
