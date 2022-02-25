# T-Brain_2021Winter
玉山人工智慧公開挑戰賽2021冬季賽 - 信用卡消費類別推薦  
連結:https://tbrain.trendmicro.com.tw/Competitions/Details/18

作法介紹影片連結
https://medium.com/@s950375/t-brain-%E4%BF%A1%E7%94%A8%E5%8D%A1%E6%B6%88%E8%B2%BB%E9%A1%9E%E5%88%A5%E6%8E%A8%E8%96%A6-%E4%BD%9C%E6%B3%95%E5%88%86%E4%BA%AB-2e482b721c19

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
