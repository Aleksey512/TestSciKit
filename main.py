import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.dummy import DummyRegressor, StrOptions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_consistent_length, check_array


class FracSumDummyRegressor(DummyRegressor):
    """
    Extended DummyRegressor with a new strategy "fracsum."

    This regressor is an extension of scikit-learn's DummyRegressor, providing an additional
    strategy, "fracsum," which returns the sum of fractional parts of the target values.

    Parameters
    ----------
    strategy : {"mean", "median", "quantile", "constant", "fracsum"}, default="mean"
        Strategy to use to generate predictions.

        * "mean": always predicts the mean of the training set
        * "median": always predicts the median of the training set
        * "quantile": always predicts a specified quantile of the training set, provided with the quantile parameter.
        * "constant": always predicts a constant value that is provided by the user.
        * "fracsum": returns the sum of fractional parts of the target values.

    constant : int or float or array-like of shape (n_outputs,), default=None
        The explicit constant as predicted by the "constant" strategy. This parameter is useful only for the "constant" strategy.

    quantile : float in [0.0, 1.0], default=None
        The quantile to predict using the "quantile" strategy. A quantile of 0.5 corresponds to the median, while 0.0 to the minimum and 1.0 to the maximum.

    fracsum : None, default=None
        Parameter specific to "fracsum" strategy. Ignored for other strategies.

    Attributes
    ----------
    constant_ : ndarray of shape (1, n_outputs)
        Mean or median or quantile of the training targets or constant value given by the user.

    n_outputs_ : int
        Number of outputs.

    See Also
    --------
    DummyClassifier: Classifier that makes predictions using simple rules.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.dummy import DummyRegressor
    >>> X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> y = np.array([0.5, 1.3, -0.8])
    >>> model = FracSumDummyRegressor(strategy="fracsum")
    >>> model.fit(X, y)
    FracSumDummyRegressor(strategy='fracsum')
    >>> model.predict(X)
    array([ 0.,  0.3, -0.8])
    >>> model.score(X, y)
    1.0
    """

    _parameter_constraints = DummyRegressor._parameter_constraints.copy()
    _parameter_constraints["strategy"] = [
        StrOptions({"mean", "median", "quantile", "constant", "fracsum"})
    ]

    def __init__(self, *, strategy="mean", constant=None, quantile=None):
        super().__init__(strategy=strategy, constant=constant, quantile=quantile)

    def fit(self, X, y, sample_weight=None):
        """
        Fit the FracSumDummyRegressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.

        y : array-like of shape (n_samples,)
            The target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        if self.strategy == "fracsum":
            y = check_array(y, ensure_2d=False, input_name="y")
            if len(y) == 0:
                raise ValueError("y must not be empty.")

            if y.ndim == 1:
                y = np.reshape(y, (-1, 1))
            self.n_outputs_ = y.shape[1]

            check_consistent_length(X, y, sample_weight)

            self.constant_ = np.sum(np.mod(y, 1))

            self.constant_ = np.reshape(self.constant_, (1, -1))

            return self
        return super().fit(X, y, sample_weight=sample_weight)


scaler = MinMaxScaler()

diabetes = load_diabetes()
diabetes.data = scaler.fit_transform(diabetes.data)
diabetes.target = scaler.fit_transform(diabetes.target.reshape(-1, 1))

X1 = diabetes.data
y1 = diabetes.target

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

model = FracSumDummyRegressor(strategy="fracsum")
model.fit(X1_train, y1_train)

# Train
train_predictions = model.predict(X1)
train_score = model.score(X1, y1)

# TestCase
test_predict_data = diabetes.data[:3]
Y1 = np.array([0.5, 1.3, -0.8])
Y2 = np.array([5, 3, -8])

test_predictions = model.predict(test_predict_data)
test_score1 = model.score(test_predict_data, Y1)
test_score2 = model.score(test_predict_data, Y2)

print("Пример Train:")
print("Наблюдаемые значения:", y1)
print("Предсказанные значения:", train_predictions)
print("R^2 score:", train_score, "\n")

print("Пример 1:")
print("Наблюдаемые значения:", Y1)
print("Предсказанные значения:", test_predictions)
print("R^2 score:", test_score1, "\n")

print("Пример 2:")
print("Наблюдаемые значения:", Y2)
print("Предсказанные значения:", test_predictions)
print("R^2 score:", test_score2, "\n")
