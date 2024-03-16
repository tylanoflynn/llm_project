# LLM Project

## Project Task
Sentiment classification of movie reviews.

## Dataset
For this project, I used the (imdb dataset)[https://huggingface.co/datasets/imdb] from Hugging Face.

## Pre-trained Model
The pretrained model that I started with was (distilbert-base-uncased-finetuned-sst-2-english)[https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english]. From the Hugging Face website: "this model is a fine-tune checkpoint of DistilBERT-base-uncased, fine-tuned on (SST-2)[https://huggingface.co/datasets/sst2]."

## Performance Metrics
Bag-of-words tokenization with a SKLearn RandomForestClassifier (default parameters):
```
		precision    recall  f1-score   support

           0       0.83      0.84      0.83       489
           1       0.84      0.83      0.84       511

    accuracy                           0.84      1000
   macro avg       0.84      0.84      0.84      1000
weighted avg       0.84      0.84      0.84      1000
```

distilbert-base-uncased-finetuned-sst-2-english:
```

		precision    recall  f1-score   support

           0       0.88      0.95      0.91       514
           1       0.94      0.86      0.90       486

    accuracy                           0.91      1000
   macro avg       0.91      0.90      0.91      1000
weighted avg       0.91      0.91      0.91      1000
```

distilbert-base-uncased-finetuned-sst-2-english trained on IMDB dataset:
```

		precision    recall  f1-score   support

           0       0.96      0.95      0.96       527
           1       0.95      0.95      0.95       473

    accuracy                           0.95      1000
   macro avg       0.95      0.95      0.95      1000
weighted avg       0.95      0.95      0.95      1000
```

Unsurprisingly, accuracy improved in each case. It is worth noting that the increase in accuracy between base distilbert and distilbert refined on the IMDB dataset was a result of a large increase in precision when determining whether or not a sentiment is negative.

## Hyperparameters
Hyperparameter search while tuning distilbert was very difficult due to the amount of time each training run took. I tried three different values for the learning rate (1e-5, 2e-5, and 1e-4) since it was the most likely to make a difference in overall accuracy, but found that this only affected runtime. It is likely that given more time and resources to perform a more extensive hyperparameter search over more diverse hyperparameters I could have achieved substantial accuracy gains.

