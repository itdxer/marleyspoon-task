import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def tag_tokenizer(tags):
    return tags.replace(",", " ").split(" ")


def timeseries_cross_validation(df, model, n_folds, n_val_weeks):
    assert n_folds >= 1, f"n_folds={n_folds}"
    assert n_val_weeks >= 1, f"n_val_weeks={n_val_weeks}"

    year_weeks = sorted(df.year_week.unique())

    if len(year_weeks) <= n_folds * n_val_weeks:
        raise Exception(
            f"Cannot run validation since data has only {len(year_weeks)} training "
            f"samples and {n_folds * n_val_weeks} needed for validation"
        )

    errors = []

    for fold in range(n_folds):
        print(f"Running CV on fold {fold + 1} / {n_folds}")
        week_start = year_weeks[-(fold + 1) * n_val_weeks]
        week_end = year_weeks[-fold * n_val_weeks - 1]

        print(f"  Running validation on weeks {week_start} - {week_end}")

        df_train = df[df.year_week < week_start]
        df_val = df[(df.year_week >= week_start) & (df.year_week <= week_end)]

        print(f"  Number of training samples: {len(df_train)}")
        print(f"  Number of validation samples: {len(df_val)}")

        assert len(df_val.year_week.unique()) == n_val_weeks

        print("  Training the model...")
        model.fit(
            df_train,
            df_train.sales,
            # Note: sample_weight helps to add more importance to the most recent obsevations
            regressor__sample_weight=((df_train.week_day - df_train.week_day.min()).dt.days + 1) ** 4,
        )
        y_predicted = np.clip(model.predict(df_val), 0, np.inf)
        error = rmse(df_val.sales, y_predicted)

        print(f"  RMSE: {error:.2f}")
        errors.append(error)

    print(f"Average RMSE accross {n_folds} folds: {np.mean(errors):.2f}")
