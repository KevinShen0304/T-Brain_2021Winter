# T-Brain_2021Winter
玉山人工智慧公開挑戰賽2021冬季賽 - 信用卡消費類別推薦  
連結:https://tbrain.trendmicro.com.tw/Competitions/Details/18

*注意:  
Data/simple_data/tbrain_cc_training_48tags_hash_final_simple.csv  
為範例數據，僅包含前200個chid，完整資料請至競賽介面下載並取代  
(代碼尚未整理優化，見諒)

# 執行順序
1.資料前處裡/gene_train_data.py  
產生train data

2.資料前處裡/gene_other_train_data.py  
產生其他train相關data

3.save_data.py  
產生依照dt區分之data

4.train.py  
訓練dt24、24、22之模型

5.predict.py  
預測dt25之結果
