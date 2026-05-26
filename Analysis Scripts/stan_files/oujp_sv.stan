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
//  dX_{t} = \theta_X (\mu_X - X_t) dt + \sigma_t dW^X_t + J_t * dNt
//    W_t : Brownian motion
//    N_t : Poisson Process
// A stochastic model of volatility
// \sigma_t = exp (V_t)
//    dV_t = \theta_v (\mu_v - V_t)dt + \sigma_v dW^h_t
//      d W^v_t: Brownain motion
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
//    Y_{t+1} ~ Normal( \Lambda_{t+1} + \alpha * (S_t - \Lambda_t),  \sigma_t^2 * (1 - alpha^2) / (2\theta))
// -----------------------------------------------------------------------

data {
  // -- training data
  int<lower=1> T; // number of observations
  vector[T] y; // observed time series
  vector[T] time_year;
  array[T] int<lower=1, upper=7> day_of_week_index;
  array[T] int<lower=1, upper=24> hour_index;
  real<lower=0> dt; // time step

  // -- mean reversion rate prior parameters
  real theta_mean;
  real<lower=0> theta_sd;
  // -- jump process
  real mu_jump_mean;
  real<lower=0> mu_jump_sd;
  real<lower=0> sigma_jump_shape;
  real<lower=0> sigma_jump_rate;

  real p_jump;

  // -- stochastic volatility 
  real theta_v_mean;
  real<lower=0> theta_v_sd;
  real mu_v_mean;
  real<lower=0> mu_v_sd;
  real sigma_v_mean;
  real<lower=0> sigma_v_sd;

  // -- deterministic function
  real overall_mean_mu;
  real<lower=0> overall_mean_sd;
  real yearly_trend_mu;
  real<lower=0> yearly_trend_sd;
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

  // -- jump process
  real mu_jump;
  real sigma_jump;

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
  real log_v_stat_sd = sigma_v / sqrt(1 - alpha_v^2);
  real log_v_stat_sd_jump = log_v_stat_sd;

  // ---- stochastic volatility ------------------------------
  vector[T] log_V;
  // t = 1: stationary initialisation
  log_V[1] = mu_v + sigma_v / sqrt(1 - alpha_v^2) * Z_v[1];

  real log_V_ou_sd  = sigma_v * sqrt( (1 - alpha_v^2) / (2 * theta_v));
  for (t in 2:T)
    log_V[t] = mu_v + alpha_v * (log_V[t-1] - mu_v) + log_V_ou_sd * Z_v[t];

  real jacobian_term = sigma_v * sqrt( (1- alpha_v^2) / (2*theta_v) ) ;

  // -- Observation model ------------------------------------
  real alpha = exp(-theta * dt); 
  real log_y_sd_scale = 0.5 * log( (1-alpha^2) / (2*theta) ); 
  vector[T-1] y_sd = exp(log_V[2:T] + log_y_sd_scale);

  // -- Determistic mean function ----------------------------
  vector[7] day_coef = append_row(0.0, day_coef_raw);
  vector[24] hour_coef = append_row(0.0, hour_coef_raw);
}

model {
  // --- Priors ----------------------------------------------
  // -- Differenced spot-price model 
  theta ~ normal(theta_mean, theta_sd); 
  mu_jump ~ normal(mu_jump_mean, mu_jump_sd);
  sigma_jump ~ gamma(sigma_jump_shape, sigma_jump_rate);

  // -- Latent volatility model
  Z_v ~ std_normal();
  theta_v ~ normal(theta_v_mean, theta_v_sd);
  mu_v ~ normal(mu_v_mean, mu_v_sd);
  sigma_v ~ normal(sigma_v_mean, sigma_v_sd);

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
  target += normal_lpdf(log_V[1] | mu_v, log_v_stat_sd) + log(log_V_ou_sd);
  for (t in 2:T) {
    real ou_mean = mu_v + alpha_v * (log_V[t-1] - mu_v);
    target += normal_lpdf(log_V[t] | ou_mean, log_V_ou_sd) + log(log_V_ou_sd);
  }

  // ---- observation model ----
  vector[T-1] ou_mean_vec = deterministic_mean[2:T] + alpha * (y[1:(T-1)] - deterministic_mean[1:(T-1)]);
  // y[2:T] ~ normal(ou_mean_vec, y_sd);

  for (t in 1:(T-1)) {
    // real X_t = y[t] - deterministic_mean; // OU state: deviation from mean
    real sigma_total = sqrt( y_sd[t]^2 + sigma_jump^2);
    real ou_mean = deterministic_mean[t+1] + alpha * (y[t] - deterministic_mean[t]);  

    target += log_mix(
      p_jump,
      normal_lpdf(y[t+1] | ou_mean + mu_jump, sigma_total), 
      normal_lpdf(y[t+1] | ou_mean, y_sd[t])          
    );
  }

}
