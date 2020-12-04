#####
# a)
####
####
# function to simulate a randomized error in linear regression
simulate <- function(reps, seed, data, true_coef = 0:ncol(data), df = 4) {
  # set seed according to the input 'seed' and initialize input
  set.seed(seed)

  # compute 'rep' sets of coefficients
  coefs <- NULL
  for (rep in seq_len(reps)) {
    coefs <- cbind(coefs, simulate_once(data, true_coef, df))
  }
  # --> this is the only step here which really takes time*

  return(structure(coefs, seed = seed))
}

# This function estimates regression coefficients based
# on the data set with a randomized target variable
simulate_once <- function(data, true_coef, df) {
  data <- simulate_response(data, true_coef, df)
  estimate_coef(data)
  # --> # both function calls take approximately the same amount of time*
}

# Create a randomized target variable
simulate_response <- function(data, true_coef, df) {
  # Compute the design matrix of the model
  design <- model.matrix(~., data = data)
  # Compute the expectations of the prediction
  expected <- design %*% true_coef
  # Add a random error term
  data[["y"]] <- expected + rt(nrow(data), df = df)
  data

  # --> no part of the code takes especially long*
}

# Function to estimate regression coefficient for a linear model
estimate_coef <- function(data) {
  model <- lm(y ~ ., data = data)
  # --> this causes the main proportion of running time*

  unname(coef(model))
}

####
# slow-sim.R must be sourced for profvis
source("slow-sim.R")

# *checked with the following code
set.seed(232323)
observations <- 5000
covariates <- 10
testdata <- as.data.frame(
  matrix(rnorm(observations * covariates),
    nrow = observations
  )
)

profvis::profvis(
  {
    test <- simulate(reps = 500, seed = 20141028, data = testdata)
  },
  interval = 0.001
)


#####
####
# b)
####
# Faster version of simulate

# Create a randomized target variable
simulate_response_fast <- function(data, expected, df) {
  # Add a random error term
  expected + rt(nrow(data), df = df)
}

# Function to estimate regression coefficient for a linear model
estimate_coef_fast <- function(x, y) {
  model <- .lm.fit(x, y)
  return(model[["coefficients"]])
}

# This function estimates regression coefficients based
# on the data set with a randomized target variable
simulate_once_fast <- function(data, true_coef, df) {

  # Compute the design matrix of the model
  design <- cbind(rep(1, nrow(data)), as.matrix(data))

  # Compute the expectations of the prediction
  expected <- design %*% true_coef

  # Compute randomized target variable
  response <- simulate_response_fast(data, expected, df)

  # Now estimate the coefficients of the randomly generated data
  estimate_coef_fast(design, response)
}

# faster version of simulate
simulate_fast <- function(reps, seed, data, true_coef = 0:ncol(data), df = 4) {
  # set seed according to the input 'seed' and initialize input
  set.seed(seed)

  # pre-allocation: define coefs before starting the loop
  coefs <- matrix(0, nrow = ncol(data) + 1, ncol = reps)

  # compute 'rep' sets of coefficients
  for (rep in seq_len(reps)) {
    coefs[, rep] <- simulate_once_fast(data, true_coef, df)
  }

  return(structure(coefs, seed = seed))
}


# Parallelisation approaches

library(foreach)
library(doRNG)

# function to simulate a randomized error in linear regression
simulate_parallel <- function(reps,
                              seed,
                              data,
                              true_coef = 0:ncol(data),
                              df = 4) {
  # set seed according to the input 'seed' and initialize input
  set.seed(seed)

  # enable parallel computing
  doParallel::registerDoParallel(cores = parallel::detectCores() - 1)

  # define a vector of export objects
  to_be_exported <- c(
    "simulate_once",
    "simulate_response",
    "estimate_coef"
  )

  # compute coefs in parallel
  coefs <- foreach(
    rep = seq_len(reps),
    .combine = "cbind",
    .export = to_be_exported
  ) %dorng% {
    simulate_once(data, true_coef, df)
  }

  return(structure(coefs[, ], seed = seed))
}


# function to simulate a randomized error in linear regression
simulate_fast_parallel <- function(reps,
                                   seed,
                                   data,
                                   true_coef = 0:ncol(data),
                                   df = 4) {
  # set seed according to the input 'seed' and initialize input
  set.seed(seed)

  # enable parallel computing
  doParallel::registerDoParallel(cores = parallel::detectCores() - 1)

  # define a vector of export objects
  to_be_exported <- c(
    "simulate_once_fast",
    "simulate_response_fast",
    "estimate_coef_fast"
  )

  # compute coefs in parallel
  coefs <- foreach(
    rep = seq_len(reps),
    .combine = "cbind",
    .export = to_be_exported
  ) %dorng% {
    simulate_once_fast(data, true_coef, df)
  }

  return(structure(coefs[, ], seed = seed))
}

## compare all four algorithms:

# simulate is the old algorithm
# simulate_fast has been improved in its structure
# simulate_parallel uses the structure of simulate with parallelisation
# simulate_parallel uses the structure of simulate_fast with parallelisation

microbenchmark::microbenchmark(
  simulate(
    reps = 500,
    seed = 20141028,
    data = testdata
  ),
  simulate_fast(
    reps = 500,
    seed = 20141028,
    data = testdata
  ),
  simulate_parallel(
    reps = 500,
    seed = 20141028,
    data = testdata
  ),
  simulate_fast_parallel(
    reps = 500,
    seed = 20141028,
    data = testdata
  ),
  times = 1
)
