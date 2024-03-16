# llm_project

## Day 1

Project Type: Sentiment Analysis
Project Area:  Customer service
Project Focus: Assessing responsiveness of SaaS companies to customer feedback.

I've noticed a trend of SaaS companies pushing changes that are counter to the desires of their user base. Often this is just to drive up profits (i.e. changing monetization models) but sometimes it is more counter-intuitive. For example, changes that reduce the usability of the platform, removal of popular features, focusing development on features that nobody seems to want, etc. I am planning on focusing on video game companies for two reasons: 

- people on social media are very vocal about their responses to changes in video games.
- I play a lot of video games

I am particularly interested in the proportion of negative feedback that leads to companies changing gears. My stretch goal is comparing this responsiveness to the responsiveness of the companies valuation to controversy. Of course, the main reason that I chose this project was so that I could name it ASaaS (Adversarial Software as a Service)/(A Sass) Companies.

## Day 2

I found out today that we are restricted to just a few possible datasets for the project. I am probably just going to go the sentiment analysis route with the 'imdb' dataset.

## Day 3

Ethical consideration: I've swapped gears to sentiment analysis using the imdb dataset. Today I am mostly thinking about ethical considerations, which there are admittedly not that many of.

The only one that springs to mind is when prediction of a user's preferences is tied to predictions of how much a concept will sell/be watched. Specifically, there are probably groups of people whose positive preferences are a decent predictor that a show/movie concept will be a mediocre seller. People who belong to this theoretical group putting their info out there in the form of positive ratings and comments on social media would in a sense be discouraging things they enjoy from ever being made. Do they have a right to know that their data is used in this way?

Bit of a stretch, I know, but I am working with what I have! And I do see a lot of people lamenting that the shows they enjoy never get renewed. Coincidence? Perhaps.

Thoughts on tokenization:

TF-IDF strikes me as good for topic classification, but maybe not ideal for sentiment analysis. No matter how many times the word "bad" is used in the dataset, it is likely a good predictor of whether a review is positive or negative. I am going to do some basic text cleaning and BOW tokenization.

Results:

A default random forest classifier gave:

ROC_AUC_Score: 0.8505
Confusion matrix: array([[10737,  1975],
       			[ 1763, 10525]])
       			
## Day 4

Applying a pretrained model to the same data should have been extremely straightforward, but I ran into an issue where at least one of the reviews in the test set was too long for the pretrained model, even though it was originally trained on the same data. I am going to try custom tokenizing the data with a max_length of 512 (the size of the tensor b indicated in the error) before proceeding.

Pretokenizing also didn't seem to work. I will try a different model.

Different models had the same issue. The trick was specifying max_length=512 and truncate=True in the definition of the pipeline.

# Day 5

I ran the pretrained model overnight. However, it does not seem to have worked correctly. The performance is much worse than the simple encoding with the Random Forest Regressor.

ROC_AUC_Score: 0.4783078307830783
Confusion matrix: array([[450,  45],
       			[481,  24]])
       			
The test set is notably smaller, but given that it is a random sample of the original test set it is unlikely that this is the cause of the poor evaluation.

The model from the lecture gave me a better idea of what pre-trained model to use and how to evaluate it. Using 'distilbert-base-uncased-finetuned-sst-2-english' from Hugging Face and a classification report gives

		precision    recall  f1-score   support

           0       0.87      0.92      0.89       509
           1       0.91      0.86      0.88       491

    accuracy                           0.89      1000
   macro avg       0.89      0.89      0.89      1000
weighted avg       0.89      0.89      0.89      1000


Or to compare it to the scoring of the other models

ROC_AUC_Score: 0.8854588886799324
Confusion matrix: array([[466,  43],
       			[ 71, 420]])
       			
Which sadly seems to only slightly beat the original BOW to random forest pipeline I created. Clearly, some fine-tuning is necessary.

## Day 6

The fine-tuned model is currently training. One thing I have noticed about BERT and other LLM models is that they take quite a long time even to feedforward on standard hardware,
nevermind training.

I am hoping that this model will beat the accuracy of the previous ones by quite a bit, since it is the most robust model trained on the entire trainset. If not, it may just be that the task is too simple and BOW tokenization with a random forest classifier already approaches the limits of potential accuracy on the dataset.

With the following training_args:

training_args = TrainingArguments(
    output_dir='my_model',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    push_to_hub=True,
)

The model trained on the IMDB dataset achieved the following metrics on the test data:

Loss = 0.1800

Which was not very interpretable. As per yesterday's lecture, I will define a custom compute_metrics function and set the trainer's compute_metrics parameter to it.

This gave the following metrics on the test data:

{'eval_loss': 0.18005608022212982,
 'eval_accuracy': 0.93652,
 'eval_f1': 0.9365631370667947,
 'eval_precision': 0.9359271390908365,
 'eval_recall': 0.9372,
 'eval_runtime': 274.4995,
 'eval_samples_per_second': 91.075,
 'eval_steps_per_second': 5.694,
 'epoch': 1.0}

So the model did gain a great deal of accuracy after being fine-tuned for the dataset!

The final cost in terms of Google cloud compute credits was $0.85. Not bad!


