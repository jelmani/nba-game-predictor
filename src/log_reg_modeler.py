import numpy as np
import pandas as pd
from sklearn import metrics

class LogRegModeler:
    """
    Logistic Regression Modeler 
    """

    def __init__(self, alpha: float = 0.05, lambda_: int = 30, iterations: int = 500):
        self.alpha = alpha
        self.lambda_ = lambda_
        self.iterations = iterations
        # learned params (set in fit)
        self.w_: np.ndarray | None = None
        self.b_: float | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    # ---------- public API ----------

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        norm_X, self.mean_, self.std_ = self._get_scaled_matrix(X_train.to_numpy())
        self.w_, self.b_ = self._compute_gradient_descent(norm_X, y_train.to_numpy(), self.alpha, self.lambda_, self.iterations)

    def predict(self, X):
        return self._compute_model_output(self._standardize_with_fitted(X), self.w_, self.b_)
    

    def test_model(self, X, y_test):
        y_pred = self._compute_model_output(self._standardize_with_fitted(X), self.w_, self.b_)
        y_pred = y_pred.round()
        self._print_evaluation(y_test, y_pred)



    # ---------- modeling ----------
    

    def _sigmoid(self, z):
        """
        Compute the sigmoid of z

        Args:
            z (ndarray): A scalar, numpy array of any size.

        Returns:
            g (ndarray): sigmoid(z), with the same shape as z
            
        """

        g = 1/(1+np.exp(-z))
    
        return g
    
    def _compute_model_output(self, X, w, b):
        # print(f"model output shape: {self._sigmoid((np.dot(X, w) + b)).shape}")
        return self._sigmoid((np.dot(X, w) + b))


    def _get_scaled_matrix(self, X):
        mean_x = np.mean(X, axis=0)
        std_x = np.std(X, axis=0)
        scaled_X = (X - mean_x) / std_x
        return scaled_X, mean_x, std_x

    def _to_numpy_2d(self, X) -> np.ndarray:
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr.astype(float, copy=False)
    
    def _standardize_with_fitted(self, X) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        Xn = self._to_numpy_2d(X)
        return (Xn - self.mean_) / self.std_
    
    def _compute_cost(self, X, w, b, y_target, lambda_):
        f_wb = self._compute_model_output(X, w, b)
        loss = -y_target*np.log(f_wb) - (1 - y_target) * np.log(1 - f_wb) + (lambda_ / (2 * len(X))) * np.sum(w ** 2)
        return np.mean(loss)
    
    def _compute_gradient(self, X, y_target, w, b, lambda_):
        dj_dw = (np.dot((self._compute_model_output(X, w, b) - y_target), X) / len(X)) + ((lambda_ / len(X)) * w)
        dj_db = np.mean((self._compute_model_output(X, w, b) - y_target))
        return (dj_dw, dj_db)
    
    def _compute_gradient_descent(self, X, y_target, alpha, lambda_, iterations):
        w = np.zeros(X.shape[1])
        # print(f"w shape: {w.shape}")
        b = 0
        cost_history = []
        for _ in range(iterations):
            dj_dw, dj_db = self._compute_gradient(X, y_target, w, b, lambda_)
            w -= alpha * dj_dw
            b -= alpha * dj_db
            cost_history.append(self._compute_cost(X, w, b, y_target, lambda_))
        # TO DO output to a file
        # Graph the learning curve
        # plt.plot(np.arange(iterations), cost_history)
        # plt.xlabel('Iterations')
        # plt.ylabel('J(w,b)')
        # plt.title('Learning Curve')
        # plt.show()
        return w, b
    
    def _print_evaluation(self, y_test, y_pred):
        print('----------------------------------')

        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
        print("Precision:", metrics.precision_score(y_test, y_pred))
        print("Recall:", metrics.recall_score(y_test, y_pred))
        
        print('----------------------------------')
        confusionMatrix = metrics.confusion_matrix(y_test, y_pred)
        print('Confusion Matrix:')
        print(confusionMatrix)
