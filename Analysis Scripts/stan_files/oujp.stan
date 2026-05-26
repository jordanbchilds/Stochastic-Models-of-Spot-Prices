// Deterministic mean + Ornstein-Uhlenbeck + Poisson jump process
// Correct specification: modelled in levels
//
// -- Generative model ---------------------------------------------------
//
//   y_t = mu_t + X_t
//
//   Deterministic mean:
//     mu_t = mu_0 + beta_{B_t} + gamma_{C_t}
//       mu_0    : overall mean
//       beta_i  : day-of-week effect i  (corner constraint: beta_1  = 0)
//       gamma_i : hour-of-day effect i  (corner constraint: gamma_1 = 0)
//
//   OU stochastic component (zero long-run mean, since mu_t handles the mean):
//     dX_t = -theta * X_t * dt + sigma * dW_t + J_t * dN_t
//
//   OU closed-form transition given X_t = y_t - mu_t:
//     X_{t+dt} | X_t ~ Normal(alpha * X_t, sigma_eps^2)
//     alpha = exp(-theta * dt)
//     sigma_eps = sigma * sqrt((1 - exp(-2*theta*dt)) / (2*theta))
//
//   Poisson jump process, with constant rate \lambda
//     N_t ~ Poisson(\lambda * dt )
//     J_t ~ Normal(\mu_J, \sigma_J)
//
// -- Poisson jump process -----------------------------------------------
//
//   At most one jump per step (small dt approximation).
//   Jump probability per step : p_jump  (parameterised directly in [0,1])
//   Jump size ~ Normal(mu_jump, sigma_jump^2)
//   Jump noise is independent of OU diffusion noise.
//
//   Mixture likelihood per transition:
//     (1 - p_jump) * N(y_{t+1}; ou_mean, sigma_eps^2)
//   +      p_jump  * N(y_{t+1}; ou_mean + mu_jump,  sigma_eps^2 + sigma_jump^2)
//
//   where ou_mean = mu_{t+1} + alpha * (y_t - mu_t)
//

data {
  // -- training data
  int<lower=1> T;
  vector[T] y;
  array[T] int<lower=1, upper=7>  day_of_week_index;
  array[T] int<lower=1, upper=24> hour_index;
  vector[T] time_year;
  real<lower=0> dt;  // time-step size

  // -- validation data
  int<lower=0> T_fore;
  array[T_fore] int<lower=1, upper=7>  day_of_week_index_fore;
  array[T_fore] int<lower=1, upper=24> hour_index_fore;

  // -- Prior parameters
  real theta_mean;
  real<lower=0> theta_sd;
  real<lower=0> sigma_shape;
  real<lower=0> sigma_rate;

  real<lower=0> p_jump_alpha;
  real<lower=0> p_jump_beta;
  real mu_jump_mean;
  real<lower=0> mu_jump_sd;
  real<lower=0> sigma_jump_shape;
  real<lower=0> sigma_jump_rate;

  real overall_mean_mu;
  real<lower=0> overall_mean_sd;
}

transformed data {
  int N = T - 1;  // number of transitions
}

parameters {
  // -- OU parameters
  real<lower=0> theta; // mean-reversion speed
  real<lower=0> sigma; // diffusion volatility

  // -- Deterministic mean function
  real overall_mean;
  real yearly_trend;
  vector[6] day_coef_raw; // 6 free coefficients; day 1 fixed to 0
  vector[23] hour_coef_raw; // 23 free coefficients; hour 1 fixed to 0
  matrix[2,3] fourier_coef;

  // -- Jump parameters
  real<lower=0, upper=1> p_jump; // jump probability per step
  real mu_jump; // mean jump size
  real<lower=0> sigma_jump; // jump size standard deviation
}

transformed parameters {
  // OU transition coefficients
  real alpha = exp(-theta * dt);
  real sigma_eps = sigma * sqrt((1 - exp(-2 * theta * dt)) / (2 * theta));

  // Coefficient vectors with corner constraint (index 1 = 0)
  vector[7]  day_coef  = append_row(0.0, day_coef_raw);
  vector[24] hour_coef = append_row(0.0, hour_coef_raw);

  // Deterministic mean at every observed time point
  vector[T] deterministic_mean = overall_mean + yearly_trend * time_year + day_coef[day_of_week_index] + hour_coef[hour_index];
  for (i in 1:3) {
    deterministic_mean += fourier_coef[1,i] * cos(2*pi()*i*time_year) + fourier_coef[2,i] * sin(2*pi()*i*time_year);
  }
}

model {
  // -- Priors -----------------------------------------------------------
  // -- OU process
  theta ~ normal(theta_mean, theta_sd);
  sigma ~ gamma(sigma_shape, sigma_rate);
  // -- jump process
  p_jump ~ beta(p_jump_alpha, p_jump_beta);   // sparse jump prior; expect few jumps
  mu_jump ~ normal(mu_jump_mean, mu_jump_sd);
  sigma_jump ~ gamma(sigma_jump_shape, sigma_jump_rate);
  // -- mean function 
  overall_mean ~ normal(overall_mean_mu, overall_mean_sd);
  day_coef_raw ~ normal(0, 2);
  hour_coef_raw ~ normal(0, 2);
  for (i in 1:2) 
    for (j in 1:3)
      fourier_coef[i,j] ~ normal(0, 2);


  // -- Likelihood -------------------------------------------------------
  // Condition on y[1]; evaluate all N = T-1 transitions.
  // Each step is a mixture over jump / no-jump.
  //
  // log_mix(p, lp1, lp2) = log(p*exp(lp1) + (1-p)*exp(lp2))
  //   lp1 : jump component    (weight p_jump)
  //   lp2 : no-jump component (weight 1 - p_jump)

  real sigma_jump_total = sqrt(sigma_eps^2 + sigma_jump^2);

  for (t in 1:N) {
    // real X_t = y[t] - deterministic_mean; // OU state: deviation from mean
    real ou_mean = deterministic_mean[t+1] + alpha * (y[t] - deterministic_mean[t]);  // mean-reverting level prediction
    target += log_mix(
      p_jump,
      normal_lpdf(y[t+1] | ou_mean + mu_jump, sigma_jump_total), 
      normal_lpdf(y[t+1] | ou_mean, sigma_eps)          
    );
  }
}

/*
generated quantities {
  // -- In-sample one-step-ahead quantities ------------------------------
  //
  // y_expected[t] = E[y_t | y_{1:(t-1)}]  (marginal expectation over jump)
  //               = ou_mean + p_jump * mu_jump
  //
  // y_pred[t]     ~ p(y_t | y_{1:(t-1)})  (single posterior predictive draw)
  //
  // Both condition on the OBSERVED y[t] at each step (filtering, not smoothing).
  
  vector[T] jump_prob;
  // vector[T] y_pred_expected;
  // vector[T] y_pred;
  {
    jump_prob[1] = 0;
    // y_pred_expected[1] = y[1];
    // y_pred[1] = y[1];

    for (t in 1:N) {
      real ou_mean = deterministic_mean[t+1] + alpha * (y[t] - deterministic_mean[t]);

      // Marginal expectation (analytically average over jump probability)
      // y_pred_expected[t+1] = ou_mean + p_jump * mu_jump;

      // Draw from the jump mixture
      int jump = bernoulli_rng(p_jump);
      real m = ou_mean + jump * mu_jump;
      real s  = sqrt(sigma_eps^2 + jump * sigma_jump^2);

      // y_pred[t+1] = normal_rng(m, s);

      real lp_jump = log(p_jump) + normal_lpdf(y[t+1]| ou_mean + mu_jump, sqrt(sigma_eps^2 + sigma_jump^2));
      real lp_nojump = log1m(p_jump) + normal_lpdf(y[t+1] | ou_mean, sigma_eps);
      jump_prob[t+1] = exp(lp_jump - log_sum_exp(lp_jump, lp_nojump));
    }
  }
  
  // -- Out-of-sample forecast: p(y_{T+k} | y_{1:T}), k = 1..T_fore -----
  //
  // Propagates uncertainty forward via Monte Carlo:
  //   - initialise from the last observed state X_T = y[T] - deterministic_mean[T]
  //   - at each step: draw a jump, draw y_{T+k}, advance the state
  //
  // y_expected_forecast[k] = E[y_{T+k} | sampled path to T+k-1]
  //                          (conditional mean, marginalised over jump)
  // y_forecast[k]          ~ p(y_{T+k} | sampled path to T+k-1)
  //
  // Aggregating across MCMC draws gives the full posterior predictive.

  vector[T_fore] y_forecast_expected;
  vector[T_fore] y_forecast;

  {
    // Deterministic mean for the forecast period
    vector[T_fore] deterministic_mean_fore = overall_mean + day_coef[day_of_week_index_fore] + hour_coef[hour_index_fore];
    real ou_mean;
    int jump;

    // mean-reverting prediction
    ou_mean = deterministic_mean_fore[1] + alpha * (y[T] - deterministic_mean[T]); 
    
    // Marginal expected value at this horizon
    y_forecast_expected[1] = ou_mean + p_jump * mu_jump;

    // Sample from the mixture and advance the path
    jump = bernoulli_rng(p_jump);
    y_forecast[1] = normal_rng(ou_mean + jump * mu_jump, sqrt(sigma_eps^2 + jump * sigma_jump^2));

    for (t in 2:T_fore) {
      ou_mean = deterministic_mean_fore[t] + alpha * (y_forecast[t-1] - deterministic_mean_fore[t-1]); 
      y_forecast_expected[t] = ou_mean + p_jump * mu_jump;
      jump = bernoulli_rng(p_jump);
      y_forecast[t] = normal_rng(ou_mean + jump * mu_jump, sqrt(sigma_eps^2 + jump * sigma_jump^2));
    }
  }
}
*/
