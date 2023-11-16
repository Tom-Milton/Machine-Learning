import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Install latestdata from https://github.com/beoutbreakprepared/nCoV2019/tree/master/latest_data

def balanced_accuracy(y, y_pred):
    """Creates score for maximising balanced accuracy"""

    # Calculates true positive rate
    true_pos = np.where((y == 1) & (y_pred == 1), 1, 0)
    num_true_pos = np.sum(true_pos)
    true_pos_rate = num_true_pos / np.sum(y == 1)

    # Calculates true negative rate
    true_neg = np.where((y == 0) & (y_pred == 0), 1, 0)
    true_neg_pos = np.sum(true_neg)
    true_neg_rate = true_neg_pos / np.sum(y == 0)

    return (true_pos_rate + true_neg_rate) / 2


def hyperparameter_tuning(X_train, y_train, mode):
    """Performs hyperparameter tuning for given mode (model)"""

    if mode == 0:
        model = LogisticRegression()
        grid = {
            'max_iter': [1000],
            'random_state': [42],
            'C': [i for i in range(250, 750, 1)],
            'class_weight': [{0: 1, 1: w / 10} for w in range(10, 20, 1)],
        }

    if mode == 1:
        model = LinearRegression()
        grid = {
            'fit_intercept': [True, False],
            'normalize': [True, False],
        }

    if mode == 2:
        model = SVR()
        grid = {
            'C': [i for i in range(20, 40, 1)],
            'gamma': [i / 10 for i in range(1, 20, 5)],
            'epsilon': [i / 1000 for i in range(5, 50, 5)],
        }

    if mode == 3:
        model = KNeighborsRegressor()
        grid = {
            'n_neighbors': list(range(1, 50)),
            'leaf_size': list(range(1, 50)),
        }

    print('\033[95m', 'hyperparameter tuning ', model, '\033[0m')
    if mode == 0:
        score = metrics.make_scorer(balanced_accuracy)
        grid_search = GridSearchCV(estimator=model, param_grid=grid, scoring=score, cv=3, verbose=2)
    elif mode > 0:
        grid_search = GridSearchCV(estimator=model, param_grid=grid, scoring='r2', cv=3, verbose=2)
    grid_test = grid_search.fit(X_train, y_train)

    # Saves hyperparameters for given model to json
    with open('Best Hyperparameters.json', 'r+') as f:
        dict_list = json.load(f)
        dict_list[mode] = grid_test.best_params_
        f.seek(0)
        json.dump(dict_list, f)
        f.truncate()


def get_tuned_model(model, X_train, y_train, mode):
    """Retrieves a hyperparameter tuned version of the given mode (model)"""

    with open('Best Hyperparameters.json') as f:
        # Retrieves hyperparameters of given model
        dict_list = json.load(f)
        best_params = dict_list[mode]

        # If logistic regression, correct 'class_weight' parameter since its original formatting is removed by json
        if mode == 0:
            best_params['class_weight'] = {int(key): value for key, value in best_params['class_weight'].items()}

        # Apply hyperparameters to model
        tuned_model = model(**best_params)
        tuned_model.fit(X_train, y_train)

        return tuned_model


def classification_stats(model, X_test, y_test):
    """Calculates basic performance stats"""

    # Print basic stats
    y_pred = model.predict(X_test)
    print(metrics.confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred), '\n')


def plot_metrics(models, X_test, y_test, title):
    """Calculates and plots relevant regression metrics"""

    # Initialises dataframe for holding regression metrics
    metrics_df = pd.DataFrame(columns=['model', 'Max Error', 'Mean Absolute Error', 'Mean Squared Error', 'r2',
                                       'Exact Accuracy', 'Within Day Accuracy'])

    for model in models:
        # Retrieves accuracy of prediction
        y_pred = model.predict(X_test)
        distance = [round(y_pred[i] - y_test.iloc[i]) for i in range(len(y_pred))]
        exact_accuracy = distance.count(0) / len(distance)
        within_day_accuracy = sum(1 for i in distance if i in [-1, 0, 1]) / len(distance)

        # Calculates performance metrics
        me = metrics.max_error(y_test, y_pred)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        r2 = metrics.r2_score(y_test, y_pred)

        # Add to dataframe
        metrics_df.loc[len(metrics_df)] = [str(model).split('(')[0], me, mae, mse, r2,
                                           exact_accuracy, within_day_accuracy]

    # Initialise plot
    fig, axs = plt.subplots(nrows=2, ncols=3)
    fig.suptitle(title + ' Regression Metrics', size=14)

    # Plot metrics
    for i, ax in enumerate(fig.axes):
        sns.barplot(x='model', y=metrics_df.columns[i + 1], hue='model', data=metrics_df,
                    palette='Blues', dodge=False, ax=ax)

        ax.set_title(metrics_df.columns[i + 1])
        ax.set_xlabel(''), ax.set_ylabel(''), ax.set_xticks([])
        ax.get_legend().remove()

    # Format and save plot
    plt.tight_layout()
    plt.savefig(title + ' Regression Metrics.png')


def plot_learning_curves(models, X_train, y_train, title):
    """Plots the learning curve"""

    # Initialise plot
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))
    fig.suptitle(title + ' Learning Curves', size=14)
    plt.setp(axs, ylim=(0, 0.75))

    # Plot learning curve
    for i, ax in enumerate(fig.axes):
        # Retrieve learning curve data
        train_sizes, train_scores, validation_scores = learning_curve(
            models[i], X_train, y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=3, scoring='r2')

        # Create df containing our two curves
        learning_df = pd.DataFrame({'train_sizes': train_sizes,
                                    'train_scores': train_scores.mean(axis=1),
                                    'validation_scores': validation_scores.mean(axis=1)})
        learning_df = pd.melt(learning_df, ['train_sizes'])

        sns.lineplot(x='train_sizes', y='value', hue='variable', data=learning_df, ax=ax)

        ax.set_title(str(models[i]).split('(')[0])
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('r2 Score')
        ax.get_legend().remove()

    # Format and save plot
    plt.tight_layout()
    plt.savefig(title + ' Learning Curves.png')


def plot_distributions(models, X_test, y_test, title):
    """Plots the distribution of predicted vs actual labels"""

    y_pred = [i.predict(X_test) for i in models]  # Calculate predicted values
    models.append('Actual'), y_pred.append(y_test)  # Add actual values

    # Initialise plot
    fig, axs = plt.subplots(nrows=2, ncols=2)
    fig.suptitle(title + ' Model Predictions Distributions', size=14)
    plt.setp(axs, xlim=(-0.5, 15), ylim=(0, 600))
    bins = np.arange(-0.5, 15.5, 1)

    # Plot predicted distributions
    for i, ax in enumerate(fig.axes):
        sns.histplot(y_pred[i], bins=bins, ax=ax)

        ax.set_title(str(models[i]).split('(')[0])
        ax.set_xlabel('Days Before Hospitalisation')
        ax.annotate(text='mean: %0.2f' % np.mean(y_pred[i]) + '\n' +
                         'standard deviation: %0.2f' % np.std(y_pred[i]),
                    xy=(0.3, 0.7), xycoords='axes fraction', size=10)

    # Format and save plot
    plt.tight_layout()
    plt.savefig(title + ' Model Predictions Distributions.png')


def predicted_vs_actual(models, X_test, y_test, title):
    """Plots predicted vs actual values"""

    # Initialise plot
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))
    fig.suptitle(title + ' Predicted vs Actual', size=14)
    plt.setp(axs, xlim=(0, 15), ylim=(0, 15))

    # Plot predicted vs actual
    for i, ax in enumerate(fig.axes):
        y_pred = models[i].predict(X_test)
        sns.regplot(x=y_pred, y=y_test, lowess=True, x_jitter=0.5, y_jitter=0.5,
                    scatter_kws={'s': 5, 'alpha': 0.1}, ax=ax)

        ax.set_title(str(models[i]).split('(')[0])
        ax.set_xlabel('Predicted Days Before Hospitalisation')
        ax.set_ylabel('Actual Days \n Before Hospitalisation')

        # Plots perfect predictor line
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lims, lims, 'r--')

    # Format and save plot
    plt.tight_layout()
    plt.savefig(title + ' Predicted vs Actual.png')


def plot_residuals(models, X_test, y_test, title):
    """Plots residuals"""

    # Initialises plot
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 3.5))
    fig.suptitle(title + ' Residual Plots', size=14)
    plt.setp(axs, xlim=(0, 10), ylim=(-10, 10))

    # Plot residuals
    for i, ax in enumerate(fig.axes):
        y_pred = models[i].predict(X_test)
        sns.regplot(x=y_pred, y=y_test - y_pred, lowess=True, x_jitter=0.5, y_jitter=0.5,
                    scatter_kws={'s': 5, 'alpha': 0.1}, ax=ax)

        ax.set_title(str(models[i]).split('(')[0])
        ax.set_xlabel('Predicted Days Before Hospitalisation')
        ax.set_ylabel('Residual Size')

        ax.axhline(y=0, color='r', linestyle='--')

    # Format and save plot
    plt.tight_layout()
    plt.savefig(title + ' Residual Plots.png')


def print_final_results(y_pred, y_test, classification_accuracy, title):
    """Prints our models final results"""

    distance = [round(y_pred[i] - y_test.iloc[i]) for i in range(len(y_test))]
    exact_accuracy = distance.count(0) / len(distance)
    within_day_accuracy = sum(1 for i in distance if i in [-1, 0, 1]) / len(distance)

    # Calculates final accuracies
    final_exact_accuracy = classification_accuracy * exact_accuracy * 100
    final_within_day_accuracy = classification_accuracy * within_day_accuracy * 100
    print(title)
    print(round(final_exact_accuracy, 2), 'percent chance of correctly predicting time until hospitalisation')
    print(round(final_within_day_accuracy, 2), 'percent chance of correctly predicting time until hospitalisation within one day', '\n')


def main(retune_hyperparameters):
    pd.set_option('display.expand_frame_repr', False)

    # Logistic regression ----------------------------------------------------------------------------------------------

    # Retrieves cleaned classification dataset and splits into training and testing
    clas_df = pd.read_csv('cleaned classification dataset.csv')
    X = clas_df.drop(['days_before_hospitalisation_binary'], axis=1)
    y = clas_df['days_before_hospitalisation_binary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    print('X_train has shape', X_train.shape)
    print('X_test has shape', X_test.shape)

    # Retrieves training subsample to be used for hyperparameter tuning as X_train is too large
    X_val, X_not_used, y_val, y_not_used = train_test_split(X_train, y_train, train_size=0.1, random_state=42)
    print('X_val has shape', X_val.shape, '\n')

    if retune_hyperparameters:
        hyperparameter_tuning(X_val, y_val, 0)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    classification_stats(logreg, X_test, y_test)

    tuned_logreg = get_tuned_model(LogisticRegression, X_train, y_train, 0)
    classification_stats(tuned_logreg, X_test, y_test)

    # Accuracy of our classification model (hyperparameter tuned logistic regression)
    classification_accuracy = tuned_logreg.score(X_test, y_test)

    # Linear regression ------------------------------------------------------------------------------------------------

    # Retrieves cleaned regression dataset and splits into training and testing
    reg_df = pd.read_csv('cleaned regression dataset.csv')
    X = reg_df.drop(['days_before_hospitalisation'], axis=1)
    y = reg_df['days_before_hospitalisation']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    if retune_hyperparameters:
        for i in range(1, 4, 1):
            hyperparameter_tuning(X_train, y_train, i)

    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    tuned_linreg = get_tuned_model(LinearRegression, X_train, y_train, 1)

    svr = SVR()
    svr.fit(X_train, y_train)
    tuned_svr = get_tuned_model(SVR, X_train, y_train, 2)

    knr = KNeighborsRegressor()
    knr.fit(X_train, y_train)
    tuned_knr = get_tuned_model(KNeighborsRegressor, X_train, y_train, 3)

    # Retrieves final accuracy of best model (SVR)
    y_pred = tuned_svr.predict(X_test)
    print_final_results(y_pred, y_test, classification_accuracy, 'Final Results:')

    # Retrieves accuracy of best naive model (median)
    median = np.ones_like(y_test) * np.median(X_test)
    print_final_results(median, y_test, classification_accuracy, 'Median Results:')

    sns.set_theme()

    # Plots not tuned regression models performance stats
    models = [linreg, svr, knr]
    plot_metrics(models, X_test, y_test, 'Not Tuned')
    plot_learning_curves(models, X_train, y_train, 'Not Tuned')
    plot_distributions(models, X_test, y_test, 'Not Tuned')
    predicted_vs_actual(models, X_test, y_test, 'Not Tuned')
    plot_residuals(models, X_test, y_test, 'Not Tuned')

    # Plots tuned regression models performance stats
    models = [tuned_linreg, tuned_svr, tuned_knr]
    plot_metrics(models, X_test, y_test, 'Tuned')
    plot_learning_curves(models, X_train, y_train, 'Tuned')
    plot_distributions(models, X_test, y_test, 'Tuned')
    predicted_vs_actual(models, X_test, y_test, 'Tuned')
    plot_residuals(models, X_test, y_test, 'Tuned')

    plt.show()


main(False)  # Don't change hyperparameter tuning to True, it takes a very long time
