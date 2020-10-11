import datetime
import argparse

import lightgbm
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from category_encoders import OrdinalEncoder

from utils import rmse, tag_tokenizer, timeseries_cross_validation, estimate_sample_weights


CATEGORICAL_FEATURES = [
    "product_type",
    "dish_type",
    "protein_cuts",
    "heat_level",
    "difficulty",
    "preferences",
]
NUMERICAL_FEATURES = [
    "proteins",
    "number_of_ingredients_per_recipe",
    "fat",
    "carbs",
    "calories",
    "n_products",
]
TEXT_FEATURES = [
    "recipe_name",
]
TAGS_FEATURES = [
    "meta_tags",
    "carbs_content",
    "dish_types",
]
model = Pipeline([
    ("feature_preprocessor", ColumnTransformer([
        # Note: Oridnal encoding works quite well for the models based on decision trees
        ("categorical", OrdinalEncoder(handle_missing="return_nan"), CATEGORICAL_FEATURES),
        # Numerical features will be passed through the model without any changes
        ("numerical", "passthrough", NUMERICAL_FEATURES),

        ("recipe_name_tfidf", TfidfVectorizer(min_df=50, stop_words="english"), "recipe_name"),
        ("meta_tags_tfidf", TfidfVectorizer(min_df=50), "meta_tags"),
        ("carbs_content_tfidf", TfidfVectorizer(min_df=50), "carbs_content"),
    ])),
    ("regressor", lightgbm.LGBMRegressor(
        n_estimators=300,
        objective="mse",
        num_leaves=10,
        max_depth=4,
        min_child_samples=60,
        learning_rate=0.02,
        colsample_bytree=0.6,
        verbosity=-1,
        extra_trees=True,
    )),
])


def do_sanity_checks(df):
    # Sanity check, making sure that there is always constant distance between week days (7 days)
    weekdays = sorted(df["week_day"].unique())
    assert len(np.unique(np.diff(weekdays))) == 1


def read_and_clean_data(csv_file_path):
    df = pd.read_csv(csv_file_path, dtype={"year_week": str, "recipe_id": str})

    df["week_day"] = df.year_week.apply(
        # Note: additional -1 indicates that we always pick Monday as the starting day of
        # the week, otherwise logic doesn't work
        lambda year_week: datetime.datetime.strptime(year_week + "-1", "%G%V-%u")
    )
    df[TAGS_FEATURES + ["dish_type"]] = df[TAGS_FEATURES + ["dish_type"]].fillna("unk")
    df = df.merge(
        (df
            .groupby("year_week")
            .agg({"recipe_id": "count"})
            .rename(columns={"recipe_id": "n_products"})
        ),
        how="left",
        left_on="year_week",
        right_index=True,
    )

    do_sanity_checks(df)
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Training production model")
    parser.add_argument(
        "--input-train-csv",
        required=True,
        help="Path to the CSV file contains training data. e.g. /home/data/train.csv",
    )
    parser.add_argument(
        "--input-test-csv",
        required=True,
        help="Path to the CSV file contains test data. e.g. /home/data/train.csv",
    )
    parser.add_argument(
        "-nw",
        "--n-val-weeks",
        default=8,
        type=int,
        help="Number of weeks used for validation",
    )
    parser.add_argument(
        "-nf",
        "--n-cv-folds",
        default=4,
        type=int,
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

    print("Reading the train and test data...")
    df_train_clean = read_and_clean_data(args.input_train_csv)
    df_test_clean = read_and_clean_data(args.input_test_csv)

    print("Running cross validation...")
    timeseries_cross_validation(df_train_clean, model, n_folds=args.n_cv_folds, n_val_weeks=args.n_val_weeks)

    print("Training the model on all training data...")
    model.fit(df_train_clean, df_train_clean.sales, regressor__sample_weight=estimate_sample_weights(df_train_clean))

    print("Saving trained production model...")
    dump(model, args.output_model)
    print(f"Saved in {args.output_model}")

    print("Generating predictions for the test data...")
    y_predicted = np.clip(model.predict(df_test_clean), 0, np.inf).astype(int)
    df_test_results = pd.DataFrame({
        "recipe_id": df_test_clean.recipe_id,
        "year_week": df_test_clean.year_week,
        "predicted_sales": y_predicted,
    })
    df_test_results.to_csv(args.output_test_pred, index=None)
    print(f"Saved predictions in {args.output_test_pred}")
