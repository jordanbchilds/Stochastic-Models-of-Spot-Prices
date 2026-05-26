// deterministic mean + OU + Poisson Jump  
// Orstein-Ulhenbeck process within an independent Poisson jump process and deterministic mean function
// 
// -- Determinist mean function ----------------------------------------
//  \Lambda_t = mu_0 + \beta_{B_t} + \gamma_{C_t}
//    \mu_0 : overall mean
//    \beta_j : day j
//    \gamma_k : hour k
//    
//    B_t : day indicator for observation t
//    C_t : hour indicator for observation t
//
// -- mean-reverting process -------------------------------------------
// The stochastic model of differences
//  dX_{t} = \theta_X (\mu_X - X_t) dt + \sigma_t dW^X_t
//    W_t : Brownian motion
// A stochastic model of volatility
// \sigma_t = exp (V_t)
//    dV_t = \theta_v (\mu_v - V_t)dt + \sigma_v dW^h_t + J_v * dN_t
//      N_t : Poisson process, constant rate \lambda
//      J_v ~ Normal(mu_J, sigma_J^2)
//
// -- Model ------------------------------------------------------------
// The spot price
//  Y_t = \Lambda_t + X_t
//    Y_t : spot price at time t
//    \Lambda_t : deterministic mean function
//    X_t : stochastic process
//
// -- Model likelihood --------------------------------------------------
//  The OU process has a closed form transition density
//    \alpha = exp(-\theta * dt) 
//    \sigma_t = exp(V_t)
//    Y_{t+1} ~ Normal( \Lambda_{t+1} + \alpha * (S_t - \Lambda_t),  \sigma_t^2 (1 - alpha^2) / (2\theta))
// -----------------------------------------------------------------------

data {
  // -- training data
  int<lower=1> T; // number of observations
  vector[T] y; // observed time series
  vector[T] time_year;
  array[T] int<lower=1, upper=7> day_of_week_index;
  array[T] int<lower=1, upper=24> hour_index;
  real<lower=0> dt; // time step

  // -- validation data

  // -- mean reversion rate prior parameters
  real theta_mean;
  real<lower=0> theta_sd;

  // -- stochastic volatility 
  real theta_v_mean;
  real<lower=0> theta_v_sd;
  real mu_v_mean;
  real<lower=0> mu_v_sd;
  real sigma_v_mean;
  real<lower=0> sigma_v_sd;

  // -- probability jump p = \lambda * \delta t
  real<lower=0> p_jump_alpha;
  real<lower=0> p_jump_beta;  

  // -- jump-size mean
  real mu_jump_mean;
  real<lower=0> mu_jump_sd;

  // -- jump-size variance 
  real<lower=0> sigma_jump_shape;
  real<lower=0> sigma_jump_rate;

  // -- deterministic function
  real overall_mean_mu;
  real<lower=0> overall_mean_sd;
  real yearly_trend_mu;
  real<lower=0> yearly_trend_sd;

  // -- Parameter fixing
  real p_jump;
}

parameters {
  // -- OU parameters -----------------------------------------
  real<lower=0> theta; // mean reversion speed

  // -- SV parameters ----------------------------------------
  vector[T] Z_v;

  // -- OU process
  real<lower=0> theta_v;
  real mu_v;
  real<lower=0> sigma_v;

  // -- Jump process
  // real<lower=0, upper=1> p_jump;  
  real mu_jump;
  real<lower=0> sigma_jump;

  // -- Mean function parameters -----------------------------
  real overall_mean;
  real yearly_trend;
  vector[6] day_coef_raw;
  vector[23] hour_coef_raw;
  matrix[2,3] fourier_coef;
}

transformed parameters {
  // -- Latent process ---------------------------------------
  // ---- parameters ----
  real alpha_v = exp(-theta_v * dt);

  // ---- stochastic volatility ------------------------------

  vector[T] log_V;
  // t = 1: stationary initialisation
  log_V[1] = mu_v + sigma_v / sqrt(1 - alpha_v^2) * Z_v[1];

  real log_V_ou_sd  = sigma_v * sqrt( (1 - alpha_v^2) / (2 * theta_v));
  for (t in 2:T) {
    log_V[t] = mu_v + alpha_v * (log_V[t-1] - mu_v) + log_V_ou_sd * Z_v[t];
  }

  real jacobian_term = sigma_v * sqrt( (1- alpha_v^2) / (2*theta_v) ) ;

  // -- Observation model ------------------------------------
  real alpha = exp(-theta * dt); 
  real log_y_sd_scale = 0.5 * log( (1-alpha^2) / (2*theta) ); 

  // -- Determistic mean function ----------------------------
  vector[7] day_coef = append_row(0.0, day_coef_raw);
  vector[24] hour_coef = append_row(0.0, hour_coef_raw);
}

model {
  // --- Priors ----------------------------------------------
  // -- Differenced spot-price model 
  theta ~ normal(theta_mean, theta_sd); 

  // -- Latent volatility model
  Z_v ~ std_normal();
  theta_v ~ normal(theta_v_mean, theta_v_sd);
  mu_v ~ normal(mu_v_mean, mu_v_sd);
  sigma_v ~ normal(sigma_v_mean, sigma_v_sd);

  // ---- volatility jump process  
  // p_jump ~ beta(p_jump_alpha, p_jump_beta); 
  mu_jump ~ normal(mu_jump_mean, mu_jump_sd);
  sigma_jump ~ gamma(sigma_jump_shape, sigma_jump_rate);

  // -- Deterministic mean function
  overall_mean ~ normal(overall_mean_mu, overall_mean_sd);
  yearly_trend ~ normal(yearly_trend_mu, yearly_trend_sd);
  day_coef_raw ~ normal(0, 1);
  hour_coef_raw ~ normal(0, 1);
  for (i in 1:2) 
    for (j in 1:3)
      fourier_coef[i,j] ~ normal(0, 2);

  // -- Model -------------------------------------------------
  // ---- deterministic mean ----
  vector[T] deterministic_mean = overall_mean + yearly_trend * time_year + day_coef[day_of_week_index] + hour_coef[hour_index];
  for (i in 1:3) {
    deterministic_mean += fourier_coef[1,i] * cos(2*pi()*i*time_year) + fourier_coef[2,i] * sin(2*pi()*i*time_year);
  }
  
  // ---- latent process ----
  real log_v_stat_sd = sigma_v / sqrt(1 - alpha_v^2);
  real log_v_stat_sd_jump = sqrt(log_v_stat_sd^2 + sigma_jump^2);

  target += log_mix(p_jump, 
    normal_lpdf(log_V[1] | mu_v + mu_jump, log_v_stat_sd_jump),
    normal_lpdf(log_V[1] | mu_v, log_v_stat_sd)
  ) + log(log_v_stat_sd);

  real log_v_ou_sd = sigma_v * sqrt( (1 - alpha_v^2) / (2*theta_v) );
  real log_v_ouj_sd = sqrt( log_v_ou_sd^2 + sigma_jump^2);

  for (t in 2:T) {
    real ou_mean = mu_v + alpha_v * (log_V[t-1] - mu_v);
    target += log_mix(p_jump, 
      normal_lpdf(log_V[t] | ou_mean + mu_jump, log_v_ouj_sd),
      normal_lpdf(log_V[t] | ou_mean, log_v_ou_sd)
    )  + log(log_v_ou_sd);
  }

  // ---- observation models ----
  vector[T-1] ou_mean_vec = deterministic_mean[2:T] + alpha * (y[1:(T-1)] - deterministic_mean[1:(T-1)]);
  vector[T-1] y_sd = exp(log_V[2:T] + log_y_sd_scale);
  y[2:T] ~ normal(ou_mean_vec, y_sd);
}

/*
generated quantities {
  vector[T_fore] deterministic_mean_fore = overall_mean  + day_coef[day_of_week_index_fore] + hour_coef[hour_index_fore];
  //
  // -- within-sample prediction --------------------------------------
  //
  vector[T] y_pred;
  vector[T] y_pred_expected;
  real final_log_V;
  {
    vector[T] log_V_pred;
    int jump_ind;

    jump_ind = bernoulli_rng(p_jump);

    log_V_pred[1] = normal_rng(mu_v + jump_ind * mu_jump, sqrt(sigma_v^2 / (1 - alpha_v^2) + jump_ind * sigma_jump^2));

    y_pred[1] = y[1];
    y_pred_expected[1] = y[1];

    vector[T-1] ou_mean_vec = deterministic_mean[2:T] + alpha * (y[1:(T-1)] - deterministic_mean[1:(T-1)]);
    real log_v_var = sigma_v^2 * (1 - alpha_v^2) / (2*theta_v);
    real ou_scale = sqrt( (1-alpha^2) / (2*theta) );
    for (t in 2:T) {
      jump_ind = bernoulli_rng(p_jump);
      real log_v_ou_mean = mu_v + alpha_v * (log_V_pred[t-1] - mu_v);
      log_V_pred[t] = normal_rng(log_v_ou_mean + jump_ind * mu_jump, sqrt(log_v_var + jump_ind * sigma_jump^2));
      y_pred_expected[t] = ou_mean_vec[t-1];
      y_pred[t] = normal_rng(ou_mean_vec[t-1], exp(log_V_pred[t]) *  ou_scale);
    }
    final_log_V = log_V_pred[T];
  }

  //
  // -- out-of-sample prediction ---------------------------------------------
  //
  vector[T_fore] y_forecast;
  vector[T_fore] y_forecast_expected;
  {
    vector[T_fore] log_V_fore;
    int jump_ind;
    real log_v_ou_mean;
    real log_v_ou_var;
    real prev_value;
    real prev_mean;

    jump_ind = bernoulli_rng(p_jump);
    
    log_v_ou_mean = mu_v + alpha_v * (final_log_V - mu_v);
    log_v_ou_var = sigma_v^2 * (1 - alpha_v^2) / (2*theta_v);
    log_V_fore[1] = normal_rng(log_v_ou_mean + jump_ind * mu_jump, sqrt(log_v_ou_var + jump_ind * sigma_jump^2));
    
    prev_value = y[T];
    prev_mean = deterministic_mean[T];
    real ou_sd_scale = sqrt((1 - alpha^2) / (2 * theta));

    y_forecast_expected[1] = deterministic_mean_fore[1] + alpha * (prev_value - prev_mean);
    y_forecast[1] = normal_rng(y_forecast_expected[1], exp(log_V_fore[1]) * ou_sd_scale);

    prev_value = y_forecast[1];
    prev_mean = deterministic_mean_fore[1];
    for (t in 2:T_fore) {
      jump_ind = bernoulli_rng(p_jump);

      log_v_ou_mean = mu_v + alpha_v * (log_V_fore[t-1] - mu_v);
      log_v_ou_var = sigma_v^2 * (1 - alpha_v)^2 / (2*theta_v);

      log_V_fore[t] = normal_rng(log_v_ou_mean + jump_ind*mu_jump, sqrt(log_v_ou_var + jump_ind*sigma_jump^2));

      real ou_mean = deterministic_mean_fore[t] + alpha * (prev_value - prev_mean);
      y_forecast_expected[t] = ou_mean;
      y_forecast[t] = normal_rng(ou_mean, exp(log_V_fore[t])*ou_sd_scale);

      prev_mean = deterministic_mean_fore[t];
      prev_value = y_forecast[t];
    }
  }
}
*/
