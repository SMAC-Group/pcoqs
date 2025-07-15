library(testthat)
library(nnet)

test_that("noisy_rc returns non-negative integers", {
  set.seed(42)
  D <- sort(runif(1000))
  sigma <- 1.0
  out <- noisy_rc(c(0.2, 0.8), D, sigma)
  expect_type(out, "integer")
  expect_gte(out, 0)
})

test_that("priv_quant approximates quantile with DP noise", {
  set.seed(42)
  D <- sort(runif(1000))
  q_dp <- priv_quant(D, alpha = 0.1, rho = 1)
  q_true <- quantile(D, probs = 0.9)
  expect_true(abs(q_dp - q_true) < 0.2)
})

test_that("pcoqs works on regression", {
  set.seed(123)
  n <- 100; p <- 4
  X <- matrix(rnorm(n * p), ncol = p)
  beta <- runif(p, -2, 2)
  Y <- X %*% beta + rnorm(n)

  index_cal <- sample(1:n, 70)
  index_test <- setdiff(1:n, index_cal)
  X_cal <- X[index_cal, ]; Y_cal <- Y[index_cal]
  X_test <- X[index_test, ]

  model <- lm(Y ~ ., data = data.frame(Y = Y_cal, X_cal))
  result <- pcoqs(model, X_cal, Y_cal, X_test, rho = 1.0)

  expect_true("output" %in% names(result))
  expect_equal(nrow(result$output), nrow(X_test))
  expect_true(all(c("lower", "upper", "prediction") %in% names(result$output)))
})

test_that("pcoqs works on multiclass classification", {
  set.seed(456)
  n <- 200; p <- 3
  X <- matrix(rnorm(n * p), ncol = p)
  logits <- X[, 1:3]
  probs <- t(apply(logits, 1, function(l) exp(l) / sum(exp(l))))
  Y <- apply(probs, 1, function(p) sample(1:3, 1, prob = p))

  index_cal <- sample(1:n, 140)
  index_test <- setdiff(1:n, index_cal)
  X_cal <- X[index_cal, ]; Y_cal <- Y[index_cal]
  X_test <- X[index_test, ]

  model <- multinom(as.factor(Y_cal) ~ ., data = data.frame(Y = as.factor(Y_cal), X_cal), trace = FALSE)

  pred_probs <- function(model, X, ...) predict(model, newdata = data.frame(X), type = "probs")
  prob_score <- function(y, probs) sapply(1:length(y), function(i) 1 - probs[i, y[i]])
  set_output <- function(probs, q_hat) {
    sets <- apply(probs, 1, function(p) {
      labs <- which(p >= 1 - q_hat)
      paste(sort(labs), collapse = ",")
    })
    data.frame(predicted_set = sets)
  }

  result <- pcoqs(
    model = model,
    X_cal = X_cal,
    Y_cal = Y_cal,
    X_test = X_test,
    alpha = 0.1,
    rho = 1.0,
    predict_fun = pred_probs,
    score_fun = prob_score,
    output_fun = set_output,
    lower_bound = 0,
    upper_bound = 1
  )

  expect_true("predicted_set" %in% names(result$output))
  expect_equal(nrow(result$output), nrow(X_test))
})
