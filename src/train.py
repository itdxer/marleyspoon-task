import datetime
import argparse

import lightgbm
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from category_encoders.ordinal import OrdinalEncoder

from utils import rmse, tag_tokenizer, timeseries_cross_validation


CATEGORICAL_FEATURES = [
    "product_type",
    "course_type",
    "dish_type",
    "protein_cuts",
    "heat_level",
]
NUMERICAL_FEATURES = [
    "proteins",
    "number_of_ingredients_per_recipe",
    "fat",
    "carbs",
    "calories",
    # "count",
]
TEXT_FEATURES = ["recipe_name"]
TAGS_FEATURES = [
    "protein_types",
    "meta_tags",
    "carbs_content",
    "cuisine",
]
model = Pipeline([
    ("feature_preprocessor", ColumnTransformer([
        # Note: Oridnal encoding works quite well for the models based on decision trees
        ("categorical", OrdinalEncoder(handle_missing="return_nan"), CATEGORICAL_FEATURES),
        # Numerical features will be passed through the model without any changes
        ("numerical", "passthrough", NUMERICAL_FEATURES),
        ("recipe_name_tfidf", TfidfVectorizer(min_df=50, stop_words="english"), "recipe_name"),

        # ("protein_types_tfidf", TfidfVectorizer(min_df=50, tokenizer=tag_tokenizer), "protein_types"),
        ("meta_tags_tfidf", TfidfVectorizer(min_df=40, tokenizer=tag_tokenizer), "meta_tags"),
        ("carbs_content_tfidf", TfidfVectorizer(min_df=40, tokenizer=tag_tokenizer), "carbs_content"),
        # ("cuisine_tfidf", TfidfVectorizer(min_df=30, tokenizer=tag_tokenizer), "cuisine"),
    ])),
    ("regressor", lightgbm.LGBMRegressor(
        n_estimators=300,
        num_leaves=7,
        max_depth=3,
        objective="rmse",
        learning_rate=0.02,
        colsample_bytree=0.5,
        verbosity=-1,
        extra_trees=True,
    )),
])


def do_sanity_checks(df):
    # Sanity check, making sure that there is always constant distance between week days (7 days)
    weekdays = sorted(df["week_day"].unique())
    assert len(np.unique(np.diff(weekdays))) == 1


def clean_data(df):
    df["week_day"] = df.year_week.apply(
        # Note: additional -1 indicates that we always pick Monday as the starting day of
        # the week, otherwise logic doesn't work
        lambda year_week: datetime.datetime.strptime(year_week + "-1", "%G%V-%u")
    )
    df[TAGS_FEATURES] = df[TAGS_FEATURES].fillna("")

    do_sanity_checks(df)
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Training production model")
    parser.add_argument(
        "-i",
        "--input-csv",
        required=True,
        help="Path to the CSV file contains training data. e.g. /home/data/train.csv",
    )
    parser.add_argument(
        "-nw",
        "--n-val-weeks",
        default=8,
        help="Number of weeks used for validation",
    )
    parser.add_argument(
        "-nf",
        "--n-cv-folds",
        default=4,
        help="Number of weeks used for validation",
    )
    parser.add_argument(
        "-om",
        "--output-model",
        required=True,
        help="Path to the file where trained model will be stored. e.g. /home/models/regressor.joblib",
    )
    parser.add_argument(
        "-ot",
        "--output-test-pred",
        required=True,
        help=(
            "Path to the file where predictions for the test data will be stored. "
            "e.g. /home/models/data/test_predictions.csv"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Reading the data...")
    df = pd.read_csv(args.input_csv, dtype={"year_week": str, "recipe_id": str})
    df_clean = clean_data(df)

    timeseries_cross_validation(df, model, n_folds=args.n_cv_folds, n_val_weeks=args.n_val_weeks)

    # date_start = "201936"
    # date_end = "201944"
    # #
    # # date_start = "201928"
    # # date_end = "201936"
    # #
    # # date_start = "201920"
    # # date_end = "201928"
    #
    # df_train = df_clean[df_clean.year_week <= date_start]
    # df_val = df_clean[(df_clean.year_week > date_start) & (df_clean.year_week <= date_end)]
    #
    # print(f"Number of training samples: {len(df_train)}")
    # print(f"Number of validation samples: {len(df_val)}")
    #
    # assert len(df_val.year_week.unique()) == args.n_val_weeks
    #
    #
    # print("Training the model on all of the data...")
    # model.fit(
    #     df,
    #     df.sales,
    #     # Note: sample_weight helps to add more importance to the most recent obsevations
    #     regressor__sample_weight=((df.week_day - df.week_day.min()).dt.days + 1) ** 4,
    # )
