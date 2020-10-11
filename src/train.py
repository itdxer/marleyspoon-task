import argparse


from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from category_encoders.ordinal import OrdinalEncoder

from utils import rmse


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
    "count",
    "cumcount",
]
TEXT_FEATURES = [
    "recipe_name_clean",
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
    ])),
    ("classifier", lightgbm.LGBMRegressor(
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


def run_sanity_checks(df):
    # Sanity check, making sure that there is always constant distance between week days (7 days)
    weekdays = sorted(df["week_day"].unique())
    assert len(np.unique(np.diff(weekdays))) == 1


def clean_data(df):
    df["week_day"] = df.year_week.apply(
        # Note: additional -1 indicates that we always pick Monday as the starting day of
        # the week, otherwise logic doesn't work
        lambda year_week: datetime.datetime.strptime(year_week + '-1', "%G%V-%u")
    )
    run_sanity_checks(df)
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
        "-n",
        "--n-val-weeks",
        default=8,
        help="Number of weeks used for validation",
    )
    parser.add_argument(
        "-om",
        "--output-model",
        required=True,
        help="Path to the file where trained model will be stored. e.g. /home/models/classifier.joblib",
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

    df = pd.read_csv(args.input_csv, dtype={"year_week": str, "recipe_id": str})
    df_clean = clean_data(df)

    date_start = "201936"
    date_end = "201944"

    df_train = df_clean[df_clean.year_week <= date_start]
    df_val = df_clean[(df_clean.year_week > date_start) & (df_clean.year_week <= date_end)]

    assert len(df_val.year_week.unique()) == args.n_val_weeks

    model.fit(
        df_train,
        df_train.sales,
        # Note: sample_weight helps to add more importance to the most recent obsevations
        sample_weight=((df_train.week_day - df_train.week_day.min()).dt.days + 1) ** 4,
    )
    y_predicted = np.clip(model.predict(df_val), 0, np.inf)
    print(f"RMSE: {rmse(df_val.sales, y_predicted)}")
