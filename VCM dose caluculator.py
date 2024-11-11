#VCM dose caluculator (grid search and randomforest), 過去の投与歴なし、一回/1日投与量
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
import numpy as np
from google.colab import files
pd.options.display.max_columns = None
data = pd.read_csv("/content/drive/MyDrive/vancomycin-data2.csv") #dataのロード
display(data) #dataの表示
parameter = data[["BW","BMI","age","Ccr"]] #patameterの抽出
loading = data[["loading"]] #試験室推奨loadingの抽出
maintain = data[["maintain"]] #試験室推奨の維持量の抽出 
TDM = data[["TDM"]]
information = data[["sex","id","初回TDMでの累積投与回数","exclusion","ARC risk factor","eGFR"]]

#上までがデータのロード、以下でAI学習操作
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold as KFold
from sklearn.model_selection import train_test_split
estimators = [10,20,40,80,160] #条件検討パラメータの一つ、n_estimator
depth = [2,4,8,16,32] #条件検討パラメータのもう一つ, max_depth
kfold = KFold(n_splits=5)
grid_paramator = {"n_estimators":estimators, "max_depth":depth}
grid_search1 = GridSearchCV(rfc(random_state=1), grid_paramator,cv=kfold) #gridsearchを実装。randomforest, gridパラメータ(上記)そしてcross validationはkfold
grid_search2 = GridSearchCV(rfc(random_state=1), grid_paramator,cv=kfold) #なぜかgridsearchが一つだと以下のgrid_loadingとgrid_maintainが全く同じになるので二つ用意
parameter_train, parameter_test, loading_train, loading_test, maintain_train, maintain_test,TDM_train,TDM_test, information_train, information_test = train_test_split(parameter, loading, maintain,TDM, information,test_size=0.2,random_state = 28
                                                                                                                                                                       
                                                                                                                                                                       )
grid_loading = grid_search1.fit(parameter_train, loading_train) #loading doseに対するgridsearch実行
grid_maintain = grid_search2.fit(parameter_train, maintain_train) #維持量に対するgridsearch実行
pred_loading = pd.Series(grid_loading.predict(parameter_test))
pred_maintain = pd.Series(grid_maintain.predict(parameter_test))

#heatmapのplot領域設定
import matplotlib.pyplot as plt
import seaborn as sns
fig, axes = plt.subplots(ncols=2, figsize=(7,4)) #1行2列のplot領域を用意

#まずはloading
print("best score for loading:{}".format(grid_loading.best_score_)) #最もscoreが高かった時のn_estimators, max_depth
print("best parameters for loading prediction:{}".format(grid_loading.best_params_))
loading_results = pd.DataFrame(grid_loading.cv_results_)
display(loading_results) #n_estimators, max_depthごとの値を表示
loading_scores= np.array(loading_results.mean_test_score).reshape(5,5) #loadingの結果をheatmap用の5×5サイズに変換
loading_scores_heatmap=sns.heatmap(loading_scores, ax=axes[0], xticklabels = estimators, yticklabels = depth, annot=True, cmap='Greys', vmin = 0.3, vmax=1.0)
loading_scores_heatmap.invert_yaxis() #普通に表示するとy軸の向きが逆だったのでinvertする
axes[0].set_title("loading dose") #heatmapのタイトル
axes[0].set_xlabel("n_estimator") #xラベル
axes[0].set_ylabel("max_depth") #yラベル

#次は維持量
print("best score for maintain:{}".format(grid_maintain.best_score_)) #最もscoreが高かった時のn_estimators, max_depth
print("best parameters for maintain prediction:{}".format(grid_maintain.best_params_))
maintain_results = pd.DataFrame(grid_maintain.cv_results_)
display(maintain_results)#n_estimators, max_depthごとの値を表示
maintain_scores= np.array(maintain_results.mean_test_score).reshape(5,5) #維持用量の結果をheatmap用の5×5サイズに変換
maintain_scores_heatmap=sns.heatmap(maintain_scores, ax=axes[1], xticklabels = estimators, yticklabels = depth, annot=True, cmap='Greys', vmin = 0.3, vmax = 1.0)
maintain_scores_heatmap.invert_yaxis() #普通に表示するとy軸の向きが逆だったのでinvertする
axes[1].set_title("maintenance dose") #heatmapのタイトル
axes[1].set_xlabel("n_estimator") #xラベル
axes[1].set_ylabel("max_depth") #yラベル


#最後にloading dose, 維持量それぞれに対しbestなmax_depth, n_estimatorsを用いてrandom forestした結果の一覧を表示
#heatmap表示
print("\n\n GridSearchの結果")
plt.subplots_adjust(wspace=0.6)
plt.show()
#fig.savefig("heatmap.jpg")
#files.download("heatmap.jpg") 
print("\n\n") #見やすくするため2行改行
test_list = pd.concat([parameter_test, loading_test, maintain_test,TDM_test, information_test], axis=1) #test dataを一つにまとめる
test_list_reset = test_list.reset_index() #indexを1からふりなおす(最初は患者indexのままのため)。やらないと以下のconcatでバグる
whole_result = pd.concat([test_list_reset,pred_loading, pred_maintain], axis=1)
whole_result_change = whole_result.rename(columns={0:"pred_loading", 1:"pred_maintain_dose"})#予測値のカラム名がデフォルトで0と1なので変換
print("testサンプルに対する試験室推奨量とAI推奨量")
display(whole_result_change) #試験室の推奨値とAIの推奨値を並べる
train = pd.concat([parameter_train,loading_train,maintain_train,TDM_train,information_train],axis=1) #trainでパラメータとTDM連結
test = pd.concat([parameter_test,loading_test,maintain_test,TDM_test,information_test],axis=1) #testでパラメータとTDM連結
whole_result_change.to_csv("/content/drive/MyDrive/VCM-result.csv", index=False) #結果をgoogle driveに保存。to_csv(path+ファイル名)
train.to_csv("/content/drive/MyDrive/VCM_train.csv", index=False) #training setをgoogle driveに保存。
test.to_csv("/content/drive/MyDrive/VCM_test.csv", index=False) #test setをgoogle driveに保存。
from sklearn.metrics import accuracy_score
print("accuracy score for loading dose:{}".format(accuracy_score(loading_test, pred_loading)))
print("accuracy score for maintain dose:{}".format(accuracy_score(maintain_test, pred_maintain)))
print("\n\n") #見やすくするため2行目改行

#loading、維持量の決定に重要な特徴量を表示
print("loading用量決定における特徴量寄与度\n BW:{}\n BMI：{}\n age：{}\n Ccr：{}\n".format(*grid_loading.best_estimator_.feature_importances_.tolist())) #tolist()でnparray形式からlistに。一番手前の*は必須!
print("維持用量決定における特徴量寄与度\n BW:{}\n BMI：{}\n age：{}\n Ccr：{} \n".format(*grid_maintain.best_estimator_.feature_importances_.tolist()))