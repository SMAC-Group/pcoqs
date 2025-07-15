#' @title Noisy Range Count
#' @description Computes the count of values within a specified range with added Gaussian noise for differential privacy.
#' @param bounds A numeric vector of length 2 specifying the interval [a, b].
#' @param data A numeric vector of sorted data values.
#' @param sigma Standard deviation of the Gaussian noise.
#' @return A non-negative integer: the noisy count.
#' @export
#' @examples
#' set.seed(42)
#' data <- sort(runif(1000))
#' noisy_rc(c(0.2, 0.5), data, sigma = 1.0)
noisy_rc <- function(bounds, data, sigma) {

  noisy_count <- sum(data >= bounds[1] & data <= bounds[2]) + rnorm(1, mean = 0, sd = sigma)

  return(max(0, floor(noisy_count)))

}

#' @title Differentially Private Quantile Approximation
#' @description Approximates a (1 - alpha)-quantile using a differentially private binary search with Gaussian noise.
#' @param data A sorted numeric vector of nonconformity scores.
#' @param alpha Target error level.
#' @param rho Privacy parameter.
#' @param lower_bound Lower search bound (default 0).
#' @param upper_bound Upper search bound (default 1).
#' @param delta Convergence threshold (default 1e-10).
#' @return A numeric value: the estimated DP quantile.
#' @export
#' @examples
#' set.seed(42)
#' D <- sort(runif(1000))
#' priv_quant(D, alpha = 0.1, rho = 1.0)
priv_quant <- function(data, alpha, rho, lower_bound = 0, upper_bound = 1, delta = 1e-10) {

  n <- length(data)
  L <- upper_bound - lower_bound
  sigma <- sqrt(ceiling(log2(L / delta)) / (2 * rho))
  m <- ceiling((1 - alpha) * (n + 1))

  left <- lower_bound
  right <- upper_bound
  N <- ceiling(log2(L / delta))

  for (i in 1:N) {

    mid <- (left + right) / 2
    c <- noisy_rc(c(lower_bound, mid), data, sigma)

    if (c < m) {

      left <- mid + delta

    } else {

      right <- mid

    }

  }

  return(round((left + right) / 2, 2))

}

#' @title Differentially Private Conformal Prediction
#' @description Constructs conformal prediction outputs (intervals or sets) with differentially private quantile estimation.
#' @param model A fitted model object.
#' @param X_cal Calibration feature matrix.
#' @param Y_cal Calibration responses.
#' @param X_test Test feature matrix.
#' @param alpha Miscoverage rate.
#' @param rho Privacy parameter.
#' @param predict_fun A prediction function with arguments (model, X, ...).
#' @param score_fun Function to compute nonconformity scores: score_fun(y, y_hat).
#' @param output_fun Function to produce final output from predictions and quantile.
#' @param lower_bound Lower bound for quantile search.
#' @param upper_bound Upper bound for quantile search.
#' @param delta Convergence threshold.
#' @param predict_args List of additional arguments for prediction.
#' @return A list with prediction output and the DP quantile.
#' @export
#' @examples
#' set.seed(123)
#' X <- matrix(rnorm(100 * 3), ncol = 3)
#' Y <- X %*% c(1, -2, 1) + rnorm(100)
#' model <- lm(Y ~ ., data = data.frame(Y, X))
#' result <- p_coqs(model, X, Y, X,
#'                 alpha = 0.1, rho = 1.0)
#' head(result$output)
p_coqs <- function(model, X_cal, Y_cal, X_test,
                  alpha = 0.1, rho = 1,
                  predict_fun = function(model, X, ...) predict(model, newdata = data.frame(X), ...),
                  score_fun = function(y, y_hat) abs(y - y_hat),
                  output_fun = function(y_hat, q_hat) {
                    list(prediction = y_hat, lower = y_hat - q_hat, upper = y_hat + q_hat)
                  },
                  lower_bound = 0, upper_bound = 1, delta = 1e-10,
                  predict_args = list()) {

  Y_cal_hat <- do.call(predict_fun, c(list(model = model, X = X_cal), predict_args))

  scores <- score_fun(Y_cal, Y_cal_hat)
  scores_sorted <- sort(scores)

  q_hat <- priv_quant(scores_sorted, alpha, rho, lower_bound, upper_bound, delta)

  Y_test_hat <- do.call(predict_fun, c(list(model = model, X = X_test), predict_args))

  output_list <- output_fun(Y_test_hat, q_hat)
  output_df <- as.data.frame(output_list)

  return(list(output = output_df, quantile = q_hat))

}
