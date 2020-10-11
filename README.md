# Marley Spoon - Recipe Sales Forecast

## Problem statement

> Every week, Marley Spoon customers order recipes from the weekly menu. An important aspect of the business is that we should be able to predict the sales of recipes WoW.

More information about problem descriptions could be found in the `Data Scientist Skills Test` pdf file

## Results

1. All of the code and docker files could be found in this repository
2. The main insights about the data could be seen in the section above. EDA could be seen in the `notebooks/Data-exploration.ipynb` notebook and some information that has been discovered during the training could be seen in the `notebooks/Training.ipynb` notebook.
3. Predictions for the test data could be found in the `data/test_predictions.csv`.

## Main insights from the data

### Raw data

1. For most of the week there are 2 times less recipes compare to the number of samples. This is because pretty much each recipe in the data gets duplicated twice, once for the `product_type="2 person"` and once for `product_type="family"` (probably for more than 2 people). It's quite visible from the data that. Because recipes come in pairs, large portion of the columns have the same values, but not all of them. Those that have different might have different `calories`, `fat` or `proteins` specified which makes sense when food is being prepared for more people.
2. Demand drops around New Year. This happens probably because people either meet with their families and stop ordering food. Test data might be effected by this, but it's a bit difficult to be certain for two main reasons. First, additional activity which hasn't been observed in the data could have effected the predictions. Second, there are only 2 New Years available in the data and only one of those contains data before the New Year, so without additional business insights it might be risky to use this assumption on the test data.
3. Decrease during july 2019 might be explained based on the similar logic. People go on vacation and cancel their subscriptions. The same decrease is not visible in 2018, but it might be because number of users was growing during 2018 and drop just not visible, but that's because there is positive trend which reduces effect of the drop. There is positive trend from January 2018 until at least April 2019, but then after the summer number of subscriptions probably didn't increase back to its previous numbers.
4. Around New Year 2019 number of recipes increased which could be seen from the data. Model manages to learn the same information from data. In addition, model learns that larger values decrease sales for the individual products, which might indicate that sales were increased, but the number of users didn't change proportionally, so that demand got unchanged (e.g. user has more options, but selects fixed number of recipes)

### Predictions

Predictive model can be explored and can help to learn more about the data

1. `recipe_name` is quite important feature. There are a few words in the title that can explain increases and decreases in sales. For example, word `steak` can have large impact on the prediction, probably because people prefer steaks and there are larger sales. There are some cases where this doesn't hold and model significantly overestimates importance of this feature.
2. Like in any time dependent data the most recent events should have large impact on the future predictions. Time data has been introduced to the model implicitly, but adjusting weights of the samples. The more recent observation the more important it will be for the model.
3. Descriptions didn't show to be useful for individual predictions, but they added improvements for to the ranking model which tried to find the most/least popular choices per each week among available recipes.

## Commands

A few useful commands that can help you to reproduce results and interact with the available notebooks.

### Build docker image

From the base directory run the following command in order to build docker image

```
docker build -t marleyspoon-task .
```

### Train the final model

```
docker-compose run train-model
```

In addition to training and saving the final model into the `models` directory the command will run cross validation and it will make predictions for the test data which will be stored to the `data/test_predictions.csv` folder

### Run jupyter notebook

```
docker-compose up notebook
```
