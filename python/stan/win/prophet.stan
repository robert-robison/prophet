// Copyright (c) Facebook, Inc. and its affiliates.

// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

functions {
  real[ , ] get_changepoint_matrix(real[] t, real[] t_change, int T, int S) {
    // Assumes t and t_change are sorted.
    real A[T, S];
    real a_row[S];
    int cp_idx;

    // Start with an empty matrix.
    A = rep_array(0, T, S);
    a_row = rep_array(0, S);
    cp_idx = 1;

    // Fill in each row of A.
    for (i in 1:T) {
      while ((cp_idx <= S) && (t[i] >= t_change[cp_idx])) {
        a_row[cp_idx] = 1;
        cp_idx = cp_idx + 1;
      }
      A[i] = a_row;
    }
    return A;
  }

  // Logistic trend functions

  real[] logistic_gamma(real k, real m, real[] delta, real[] t_change, int S) {
    real gamma[S];  // adjusted offsets, for piecewise continuity
    real k_s[S + 1];  // actual rate in each segment
    real m_pr;

    // Compute the rate in each segment
    k_s[1] = k;
    for (i in 1:S) {
      k_s[i + 1] = k_s[i] + delta[i];
    }

    // Piecewise offsets
    m_pr = m; // The offset in the previous segment
    for (i in 1:S) {
      gamma[i] = (t_change[i] - m_pr) * (1 - k_s[i] / k_s[i + 1]);
      m_pr = m_pr + gamma[i];  // update for the next segment
    }
    return gamma;
  }
  
  real[] logistic_trend(
    real k,
    real m,
    real[] delta,
    real[] t,
    real[] cap,
    real[ , ] A,
    real[] t_change,
    int S,
    int T
  ) {
    real gamma[S];
    real Y[T];

    gamma = logistic_gamma(k, m, delta, t_change, S);
    for (i in 1:T) {
      Y[i] = cap[i] / (1 + exp(-(k + dot_product(A[i], delta))
        * (t[i] - (m + dot_product(A[i], gamma)))));
    }
    return Y;
  }

  // Linear trend function

  real[] linear_trend(
    real k,
    real m,
    real[] delta,
    real[] t,
    real[ , ] A,
    real[] t_change,
    int S,
    int T
  ) {
    real gamma[S];
    real Y[T];

    for (i in 1:S) {
      gamma[i] = -t_change[i] * delta[i];
    }
    for (i in 1:T) {
      Y[i] = (k + dot_product(A[i], delta)) * t[i] + (
        m + dot_product(A[i], gamma));
    }
    return Y;
  }

   // Flat trend function

    real[] flat_trend(
    real m,
    int T
  ) {
    return rep_array(m, T);
  }

  // // Seasonal Shift function

  // real[ , ] seasonal_shift(
  //   real[] delta_s,
  //   real[] t,
  //   real[ , ] B,
  //   real[] t_change2,
  //   real[ , ] X,
  //   int S2,
  //   int K,
  //   int T
  // ) {
  //   real gamma_s[S2];
  //   real Y[T];
  //   real new_X[T, K];
  //   real curr_shift;

  //   for (i in 1:T) {
  //     int j = -50;
  //     curr_shift = dot_product(B[i], delta_s);
  //     while (j < curr_shift) {
  //       j = j + 1;
  //     }
  //     new_X[i, ] = X[i + j, ];
  //   }
  //   return new_X;
  // }

  // Fourier Series function

  real [ , ] fourier_series(
    real[] t,
    int T, 
    int K,
    real period
  ) {
    real f[T, 2 * K];

    for (i in 1:T) {
      for (j in 1:K) {
        f[i, 2 * j - 1] = sin(2 * j * pi() * t[i] / period);
        f[i, 2 * j] = cos(2 * j * pi() * t[i] / period);
      }
    }
    return f;
  }

  // Construct seasonal variables

  real [ , ] seasonal_vars(
    int yearly_K,
    int weekly_K,
    real[] delta_yearly,
    real[] delta_weekly,
    real[] t_change_yearly,
    real[] t_change_weekly,
    real[] t,
    int T,
    int S_yearly,
    int S_weekly
  ) {
    real X_seasonal[T, 2 * yearly_K + 2 * weekly_K];
    real X_yearly[T, 2 * yearly_K];
    real X_weekly[T, 2 * weekly_K];
    real A_yearly[T, S_yearly];
    real A_weekly[T, S_weekly];
    real t_yearly[T];
    real t_weekly[T];

    A_yearly = get_changepoint_matrix(t, t_change_yearly, T, S_yearly);
    A_weekly = get_changepoint_matrix(t, t_change_weekly, T, S_weekly);

    for (i in 1:T) {
      t_yearly[i] = t[i] + dot_product(A_yearly[i], delta_yearly);
      t_weekly[i] = t[i] + dot_product(A_weekly[i], delta_weekly);
    }

    X_yearly = fourier_series(t_yearly, T, yearly_K, 365.25);
    X_weekly = fourier_series(t_weekly, T, weekly_K, 7);
    X_seasonal[, :(2 * yearly_K)] = X_yearly;
    X_seasonal[, (2 * yearly_K + 1):] = X_weekly;

    return X_seasonal;
  }
}

data {
  int T;                // Number of time periods
  int<lower=1> K;       // Number of regressors
  real t[T];            // Time [0-1]
  real t_days[T];            // Time in days
  real cap[T];          // Capacities for logistic trend
  real y[T];            // Time series
  int S;                // Number of changepoints
  int S_yearly;                // Number of yearly seasonal shift changepoints
  int S_weekly;                // Number of weekly seasonal shift changepoints
  real t_change[S];     // Times of trend changepoints
  real t_change_yearly[S_yearly];     // Times of yearly seasonal shift changepoints [daily]
  real t_change_weekly[S_weekly];     // Times of weekly seasonal shift changepoints [daily]
  // real X[T,K];         // Regressors
  vector[K] sigmas;     // Scale on seasonality prior
  real<lower=0> tau;    // Scale on changepoints prior
  real<lower=0> tau_yearly;    // Scale on yearly seasonal shift changepoints prior
  real<lower=0> tau_weekly;    // Scale on weekly seasonal shift changepoints prior
  int trend_indicator;  // 0 for linear, 1 for logistic, 2 for flat
  real s_a[K];          // Indicator of additive features
  real s_m[K];          // Indicator of multiplicative features
  int yearly_K;         // Number of yearly fourier terms
  int weekly_K;         // Number of weekly fourier terms
}

transformed data {
  real A[T, S];
  A = get_changepoint_matrix(t, t_change, T, S);
}

parameters {
  real k;                   // Base trend growth rate
  real m;                   // Trend offset
  real delta[S];            // Trend rate adjustments
  real delta_yearly[S_yearly];            // yearly Seasonal shift
  real delta_weekly[S_weekly];            // weekly Seasonal shift
  real<lower=0> sigma_obs;  // Observation noise
  real beta[K];             // Regressor coefficients
}

transformed parameters {
  real trend[T];
  real Y[T];
  real beta_m[K];
  real beta_a[K];
  real X_seasonal[T, 2 * yearly_K + 2 * weekly_K];    // With seasonal shifts

  if (trend_indicator == 0) {
    trend = linear_trend(k, m, delta, t, A, t_change, S, T);
  } else if (trend_indicator == 1) {
    trend = logistic_trend(k, m, delta, t, cap, A, t_change, S, T);
  } else if (trend_indicator == 2){
    trend = flat_trend(m, T);
  }

  for (i in 1:K) {
    beta_m[i] = beta[i] * s_m[i];
    beta_a[i] = beta[i] * s_a[i];
  }

  X_seasonal = seasonal_vars(yearly_K, weekly_K, delta_yearly, delta_weekly, t_change_yearly, t_change_weekly, t_days, T, S_yearly, S_weekly);

  for (i in 1:T) {
    Y[i] = (
      trend[i] * (1 + dot_product(X_seasonal[i], beta_m)) + dot_product(X_seasonal[i], beta_a)
    );
  }
}

model {
  //priors
  k ~ normal(0, 5);
  m ~ normal(0, 5);
  delta ~ double_exponential(0, tau);
  delta_yearly ~ double_exponential(0, tau_yearly);
  delta_weekly ~ double_exponential(0, tau_weekly);
  sigma_obs ~ normal(0, 0.5);
  beta ~ normal(0, sigmas);

  // Likelihood
  y ~ normal(Y, sigma_obs);
}
