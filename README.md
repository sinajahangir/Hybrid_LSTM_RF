# Hybrid_LSTM_RF
 Sample code for reproducing the results of the hybrid (LSTM-RF) model.

 ## Paper Summary

 
 This study proposes a novel hybrid method that substantially accelerates and improves deep learning (DL) model development for streamflow prediction. The method leverages a combination of a long short-term memory (LSTM) network and random forests. A hybrid encoder-decoder model is designed, where a pre-trained LSTM is utilized as an encoder to extract temporal features from the input data. Subsequently, the random forest decoder processes the encoded information to make streamflow predictions. Our method was tested on 421 catchments in the continental United States and 324 in Germany, selected from two CAMELS datasets. The hybrid method has several benefits. First, it is much more efficient than training LSTMs on each catchment individually (~14x faster). Second, it is much less computationally expensive than LSTM fine-tuning (i.e., feasible on a CPU-based workstation). Third, it achieves superior accuracy compared to a site-specific training strategy (e.g., 9.2% improvement in the median in Nash-Sutcliffe Efficiency (NSE)), competitive performance compared to regional LSTM models when trained with fewer data, and outperforms regional LSTM model by 2.73%-3.16% in median NSE. To our knowledge, this is the first decision-tree model integrated within a DL workflow to enhance fine-tuning efficiency of pre-trained models in new locations. This hybrid approach holds significant promise for future applications in hydrological modeling, particularly considering the imminent rise of geospatial foundation models in hydrology that will rely on transfer learning techniques for effective deployment.

 ## Pytorch sample codes
 The Pytorch sample code showcases how a regional LSTM model can be fine-tuned for catchments that have and have not been included in the training set. Below is an example of the results associated with CAMELS-US.

 ![hybrid_ft_random50_113_v1](https://github.com/user-attachments/assets/6919f421-7d31-4994-8cb4-726641e1dc4d)

## TensorFlow (TF) sample codes

The TF sample codes are provided to show the end-to-end model development pipeline and prediction using TF. Below is an example of the results that were obtained associated with CAMELS-DE.

<img width="659" alt="image" src="https://github.com/user-attachments/assets/e681f6e8-ae8d-48b6-a57b-bd945a81a573" />
