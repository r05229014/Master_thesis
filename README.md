# Master_thesis

這是我的碩士論文"The representation of moist convection using 3D Convolutional Neural Networks"  
## Abstract
  濕對流過程的參數化是造成氣候模式和數值天氣模式不確定性的主要原因，同時隨著氣候模式及數值天氣模式水平解析度的增加，濕對流過程該由積雲參數化處理或是由網格尺度的物理過程處理變的模糊不清(grey zone resolution)。
	  
  然而，至今仍未有研究評估在grey zone resolution下以相鄰網格作為決定對流影響效應之參數對於結果的影響。因此我的研究中利用了高解析的雲解析模式(VVM)的資料，並選用3D-CNN作為我們的模型，在grey zone resolution下以各個網格及其周邊資訊作為輸入，預測各個網格的對流效應。
	在本研究的實驗設計中建立了一個濕對流參數化的流程，讓複雜的ML-based濕對流參數化的流程有了一個完整的步驟，分別為資料的前處理、模型的選擇以及後續的驗證。
  
在驗證對流的效應時並非以簡單的RMSE或是accuracy作為我們的驗證手法，由於濕對流效應的預測並非為簡單的回歸或是分類的問題，而是複雜的結構化預測問題，單純資料上的loss其實很難在物理上分辨我們預測的好壞，因此本研究根據對流的性質來驗證模型的預測優劣。分別是對流效應的位置、對流效應的垂直分布、對流效應對水氣量的反應以及對流的垂直差異。
  
而在結果討論中本研究發現細心挑選的模型3DCNN是非常適合於預測高度複雜的對流效應的。

### 驗證
  
本研究首先挑選了四個不同濕度的環境，分別為乾到濕的時間點。
![image](https://github.com/r05229014/Master_thesis/blob/master/img/2.PNG)
  
**1. 3DCNN預測位置的驗證**
![image](https://github.com/r05229014/Master_thesis/blob/master/img/1.PNG)
  
**2. 3DCNN預測對流效應垂直統計分布的驗證**
![image](https://github.com/r05229014/Master_thesis/blob/master/img/3.PNG)
  
**3. 3DCNN預測對流效應對水氣反應的驗證**
![image](https://github.com/r05229014/Master_thesis/blob/master/img/4.PNG)
  
**4. 3DCNN預測細部垂直結構的驗證**  
![image](https://github.com/r05229014/Master_thesis/blob/master/img/5.PNG)
  
**5. 各個模型的比較**
![image](https://github.com/r05229014/Master_thesis/blob/master/img/6.PNG)
  
**6. 本研究所使用的3D-CNN架構**  
![image](https://github.com/r05229014/Master_thesis/blob/master/img/7.PNG)
