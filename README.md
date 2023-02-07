# BERT for name entity recognition

This model can tag receipt-entry like data with 3 different labels:

  1. Brand Name
  
  2. Product Type
  
  3. Specification
  
  For example: "家樂福鮮乳100ml" we'll be labeled as 
  
  家樂福 = Brand name
  
  牛奶 = Product Type
  
  100ml = Specification
  
**For trained Model:**
repository: NLP training/Model_training/model/

1. bert-base-chinese-30epoch-20batch_

2. bert-base-chinese-30epoch-64batch_ES (ES = with early stopping)


**For training codes:**
repository: NLP training/Model_training/

notebook: [Bert_tokenclassification_traing_and_testing.ipynb]
    #required packages:
        1.transformers __version__4.10.3
        2.torch __version__1.9.0
        3.datasets
        4.seqeval


**For training data:**
repository: NLP training/Training_data

numpy file: cleaned_data_np.npy
