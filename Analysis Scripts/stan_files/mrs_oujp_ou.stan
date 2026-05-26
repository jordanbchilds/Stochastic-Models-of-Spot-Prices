// deterministic mean + MRS OU process  
//  Let Y_t be the differenced spot-price data
//    Y_t = \Lambda_t + X_t
//    \Lambda_t : deterministic mean function
//    X_t : stochastic process
//
// -- Determinist mean function ----------------------------------------
//  \Lambda_t = \mu_0 + \mu_1*t + \beta_{A_t} + \gamma_{B_t} + \sum_{l=1}^3 a_l * cos( 2*pi*l*t / 8760) + b_l * sin( 2*pi*l*t / 8760)
//    \mu_0 : overall mean
//    \beta_j : day j
//    \gamma_k : hour k
//    a_l : l-th fourier `function` coeficient for cosine
//    b_l : l-th fourier `function` coeficient for sine
//    
//    A_t : day indicator for observation t
//    B_t : hour indicator for observation t
//
// -- mean-reverting process -------------------------------------------
// Base Regime R_t = 1:
//  dX_{t} = -\theta_1 X_t dt + \sigma_2 dW^1_t + J_t dN_t
//    W_t : Brownian motion
//    N_t : Poisson process with rate, \lambda
//    J_t : normally distributed, jump-size
//  
// Spike Regime R_t = 2:
//  dX_{t} = -\theta_2 X_t dt + \sigma_2 dW^2_t
//    W_t : Brownian motion
//    N_t : Poisson process with rate, \lambda
//    J_t : normally distributed, jump-size
//
// -- Jump process -----------------------------------------------------
//  dN_t : Poisson proecess with rate \lambda
//    N_{t+dt} ~ Poisson(\lambda * \delta t)
//  J_t : jump-size, independently normally distributed
//    J_t ~ Normal(\mu_J, \sigma_J^2)
//
// -- Hidden Markov states ----------------------------------------------
// The first state:
//    R_1 ~ Categorical(\pi)
//
// let R_t be the state at time t>1:
//    Pr(R_{t+dt}=j | R_t=i) = A[i,j]
//
// -- Model likelihood --------------------------------------------------
// The model likelihood can be computed via the Forward Algorithm (https://en.wikipedia.org/wiki/Forward_algorithm#:~:text=The%20forward%20algorithm%2C%20in%20the,distinct%20from%2C%20the%20Viterbi%20algorithm.)
//  The OU processes have closed form transition densities
//    let \alpha = exp(-\theta * dt), then 
//    X_{t+dt} ~ Normal( \alpha * x_t , \sigma^2 (1 - alpha^2) / (2\theta) )
//
// Regime 1: 
//  let p_jump = \lambda * dt
//    X_{t+dt} ~ (1-p_jump) N( \alpha_1 * x_t, \sigma_1^2 (1 - alpha_1^2) / (2\theta_1)) 
//                + p_jump N( \alpha_1 * x_t + \mu_J, sqrt(\sigma_1^2 (1 - alpha_1^2) / (2\theta_1) + \sigma_J^2) )
// Regime 2:
//    X_{t+dt} ~ N( \alpha_2 * x_t, \sigma_2^2 (1-alpha_2^2) / (2\theta_2) )
//
// -- Predictive quantities ----------------------------------------------
// Generated quantities:
//    Viterbi state sequence (MAP decoding)
//    Filtered state probabilities  P(z[t] = k | y[1:t])
//
// -----------------------------------------------------------------------

data {
  // -- training data
  int<lower=2> T;    // number of observations
  vector[T] y;    // observed time series
  array[T] int<lower=1, upper=7> day_of_week_index;
  array[T] int<lower=1, upper=24> hour_index;
  vector[T] time_year;
  real<lower=0> dt; 

  // -- validation data
  int<lower=1> T_fore;
  array[T_fore] int<lower=1, upper=7> day_of_week_index_fore;
  array[T_fore] int<lower=1, upper=24> hour_index_fore;

  // -- base regime prior parameters
  // ---- mean reverting process
  real<lower=0> sigma_base_shape;
  real<lower=0> sigma_base_rate;
  real<lower=0> theta_base_mu;
  real<lower=0> theta_base_sd;

  // ---- jump process 
  // real<lower=0> p_jump_alpha;
  // real<lower=0> p_jump_beta;
  real mu_jump_mean;
  real<lower=0> mu_jump_sd;
  real<lower=0> sigma_jump_shape;
  real<lower=0> sigma_jump_rate;

  // -- spike regime prior parameters
  real<lower=0> sigma_spike_shape;
  real<lower=0> sigma_spike_rate;
  real<lower=0> theta_spike_mu;
  real<lower=0> theta_spike_sd;

  // Hidden state transition prob prior parameters
  matrix[2,2] A_weights;
  vector[2] pi_weights;

  // -- Deterministi mean function
  real overall_mean_mu;
  real<lower=0> overall_mean_sd;
  real yearly_trend_mu;
  real<lower=0> yearly_trend_sd;

  real<lower=0, upper=1> p_jump;
}

parameters {
  // -- HMM structure ----------------------------------------------------------
  simplex[2] pi; // probability of first state
  array[2] simplex[2] A; // transition matrix

  // -- Stochastic model -------------------------------------------------------
  // -- OU processes
  positive_ordered[2] theta; 
  positive_ordered[2] sigma; 
  // -- Jump process
  // real<lower=0, upper=0.5> p_jump; 
  real mu_jump; 
  real<lower=0> sigma_jump; 

  // -- Deterministic mean ------------------------------------------------------
  real overall_mean;
  real yearly_trend;
  vector[6] day_coef_raw;
  vector[23] hour_coef_raw;
  matrix[2,3] fourier_coef; 
}

transformed parameters {
  // -- OU transition density ---------------------------------------------
  // ---- Mean reversion rate
  vector[2] alpha;
  alpha[1] = exp(-theta[1] * dt);
  alpha[2] = exp(-theta[2] * dt);
  // -- Conditional standard deviation
  vector<lower=0>[2] sd;
  sd[1] = sigma[1] * sqrt( -expm1(-2.0 * theta[1] * dt) / (2.0 * theta[1]) );
  sd[2] = sigma[2] * sqrt( -expm1(-2.0 * theta[2] * dt) / (2.0 * theta[2]) );
  
  // -- Deterministic mean -----------------------------------------------
  vector[7] day_coef = append_row(0.0, day_coef_raw);
  vector[24] hour_coef = append_row(0.0, hour_coef_raw);
  // Deterministic mean at every observed time point
  vector[T] deterministic_mean = overall_mean + yearly_trend * time_year + day_coef[day_of_week_index] + hour_coef[hour_index];
  for (i in 1:3) {
    deterministic_mean += fourier_coef[1,i] * cos(2*pi()*i*time_year) + fourier_coef[2,i] * sin(2*pi()*i*time_year);
  }

  // --- HMM ------------------------------------------------------------
  matrix[2, 2] log_A;
  for (j in 1:2)
    for (k in 1:2)
      log_A[j, k] = log(A[j][k]);
}

model {
  // -- Priors ----------------------------------------------------------------─
  // ---- mean function 
  overall_mean  ~ normal(overall_mean_mu, overall_mean_sd);
  yearly_trend ~ normal(yearly_trend_mu, yearly_trend_sd);
  day_coef_raw  ~ normal(0, 2);
  hour_coef_raw ~ normal(0, 2);
  for (i in 1:2) 
    for (j in 1:3)
      fourier_coef[i,j] ~ normal(0, 2);

  // ---- Regime 1: OU - JP
  theta[1] ~ normal(theta_base_mu, theta_base_sd);
  sigma[1] ~ gamma(sigma_base_shape, sigma_base_rate); // T[,10];
  // p_jump ~ beta(p_jump_alpha, p_jump_beta);
  mu_jump ~ normal(mu_jump_mean, mu_jump_sd);
  sigma_jump ~ gamma(sigma_jump_shape, sigma_jump_rate);

  // ---- Regime 2: spike state
  theta[2] ~ normal(theta_spike_mu, theta_spike_sd);
  sigma[2] ~ gamma(sigma_spike_shape, sigma_spike_rate);

  // ---- Transition rows
  A[1] ~ dirichlet([A_weights[1,1], A_weights[1,2]]'); 
  A[2] ~ dirichlet([A_weights[2,1], A_weights[2,2]]'); 
  
  // ---- First state probability
  pi ~ dirichlet([pi_weights[1], pi_weights[2]]');

  // -- Scaled Recursive algorithm (log-space) ------------------------------------------
  {
    // -- transformed parameters -------------
    // ---- Jump probability
    real log_p_jump = log(p_jump);
    real log_p_no_jump = log1m(p_jump);
    // ---- OUJ standard deviation
    real jump_sd = sqrt(sd[1]^2 + sigma_jump^2);

    // -- Forward Algorithm Components ---------------
    vector[2] log_filter_prob; // \alpha_t^k = Pr(Z_t = k | y_{1:t})
    vector[2] log_predict_prob; // log_latent_transition_prob;
    vector[2] log_component_density;
    real log_normalising_const;
    vector[2] log_emit;
    // -- Store OU process means ------
    vector[2] ou_mean; 

    ou_mean[1] = deterministic_mean[2] + alpha[1] * (y[1] - deterministic_mean[1]);  // mean-reverting level prediction
    ou_mean[2] = deterministic_mean[2] + alpha[2] * (y[1] - deterministic_mean[1]);  // mean-reverting level prediction

    // calculate p(y_1) from stationary distribution
    log_component_density[1] = log(pi[1]) + log_sum_exp(
      log_p_jump + normal_lpdf( y[2] | ou_mean[1] + mu_jump, sqrt(sigma[1]^2/ (2*theta[1]) + sigma_jump^2)),
      log_p_no_jump + normal_lpdf( y[2] | ou_mean[1] , sigma[1] / sqrt(2*theta[1]) )
    );
    log_component_density[2] = log(pi[2]) + normal_lpdf(y[2] | ou_mean[2], sigma[2] / sqrt(2*theta[2]));

    log_normalising_const = log_sum_exp(log_component_density);
    log_filter_prob[1] =  log_component_density[1] - log_normalising_const;
    log_filter_prob[2] = log_component_density[2] - log_normalising_const;

    target += log_normalising_const;

    for (t in 2:(T-1)) {
      // latent transition densities, alpha$k_{t|t-1} = Pr(Z_t = k | y_{1:t-1})
      log_predict_prob[1] = log_sum_exp(
        log_A[1,1] + log_filter_prob[1], 
        log_A[2,1] + log_filter_prob[2]
        );
      log_predict_prob[2] = log_sum_exp(
        log_A[1,2] + log_filter_prob[1], 
        log_A[2,2] + log_filter_prob[2]
        );
      
      // real X_t = y[t] - deterministic_mean[t]; // OU state: deviation from mean
      ou_mean[1] = deterministic_mean[t+1] + alpha[1] * (y[t] - deterministic_mean[t]); 
      ou_mean[2] = deterministic_mean[t+1] + alpha[2] * (y[t] - deterministic_mean[t]); 

      // obs transition densities, f(y_t | y_{1:(t-1)})
      log_emit[1] = log_sum_exp(
        log_p_jump + normal_lpdf( y[t+1] | ou_mean[1] + mu_jump, jump_sd),
        log_p_no_jump + normal_lpdf( y[t+1] | ou_mean[1], sd[1])
      );
      log_emit[2] = normal_lpdf(y[t+1] | ou_mean[2], sd[2]);
      
      // log-likelihood
      log_component_density[1] = log_predict_prob[1] + log_emit[1];
      log_component_density[2] = log_predict_prob[2] + log_emit[2];

      // latent probabilities, alpha_t
      log_normalising_const = log_sum_exp(log_component_density);
      log_filter_prob[1] = log_predict_prob[1] + log_emit[1] - log_normalising_const;
      log_filter_prob[2] = log_predict_prob[2] + log_emit[2] - log_normalising_const;

      target += log_normalising_const;
    }
  }
}

/*
generated quantities {
  vector[T_fore] deterministic_mean_fore = overall_mean  + day_coef[day_of_week_index_fore] + hour_coef[hour_index_fore];

  // -- Viterbi decoding: most probable hidden-state sequence ------------------
  array[N] int viterbi_states;
  {
    // -- Transformed parameters
    // ---- Jump probability
    real log_p_jump = log(p_jump);
    real log_p_no_jump = log1m(p_jump);
    // ---- OUJ standard deviation
    real jump_sd = sqrt(sd[1]^2 + sigma_jump^2);


    array[N] vector[2] delta; // delta[t][k] = max log-prob path to state k at t
    array[N, 2] int psi; // backpointers


    // t = 1: condition on w[1], no emission
    delta[1] = log(pi);

    // Forward Viterbi pass
    for (t in 2:N) {
      vector[2] log_emit;
      log_emit[1] = log_sum_exp(
        log_p_jump + normal_lpdf(w[t] | deterministic_mean[t] + alpha[1] * w[t-1] + mu_jump, jump_sd),
        log_p_no_jump + normal_lpdf(w[t] | deterministic_mean[t] + alpha[1] * w[t-1], sd[1])
      );
      log_emit[2] = normal_lpdf(w[t] | deterministic_mean[t] + alpha[2] * w[t-1], sd[2]);

      for (k in 1:2) {
        real s1 = delta[t-1][1] + log_A[1, k];
        real s2 = delta[t-1][2] + log_A[2, k];
        if (s1 >= s2) {
          psi[t, k]   = 1;
          delta[t][k] = s1 + log_emit[k];
        } else {
          psi[t, k]   = 2;
          delta[t][k] = s2 + log_emit[k];
        }
      }
    }

    // Backtrack
    viterbi_states[N] = (delta[N][1] >= delta[N][2]) ? 1 : 2;
    for (t in 1:(N-1))
      viterbi_states[N-t] = psi[N-t+1, viterbi_states[N-t+1]];
  }

  // -- One-step-ahead prediction -----------------------------------------------
  // y_pred : p(y_t | y_{1:t-1}) obtained by marginalising over Z_t and parameter uncertainty
  // y_expected : E[ y_t | y_{1:t-1}] obtained by marginalising over Z_t and parameter uncertainty
  // Uses *prior* state probabilities (predicted, not filtered) so it is a
  // vector[T] y_pred;
  // vector[T] y_pred_expected;
  vector[2] final_log_predict_prob; // needed for the out-of-sample forecast in next block
  // t = 1 has no predecessor, so set to the observed value
  {
    // -- transformed parameters
    // ---- log jump probability
    real log_p_jump = log(p_jump);
    real log_p_no_jump = log1m(p_jump);

    vector[2] log_filter_prob; // \alpha^k_t = Pr(Z_t = k | y_{1:t})
    vector[2] log_predict_prob; // \alpha^k_{t|t-1} = Pr(Z_t=k | y_{1:t-1})
    vector[2] log_component_density; 
    vector[2] log_emit;
    vector[2] ou_mean;
    real log_normalising_const;
    real m;
    real s;

    // condition on y_1
    // y_pred[1] = y[1];
    // y_pred_expected[1] = y[1];

    // calculate the expected values of the OU processes
    ou_mean[1] = deterministic_mean[2] + alpha[1] * (y[1] - deterministic_mean[1]); 
    ou_mean[2] = deterministic_mean[2] + alpha[2] * (y[1] - deterministic_mean[1]); 

    // Define the mean, m, and standard deviation, s, for the random draw
    if (bernoulli_rng(pi[1])) {
      int jump_draw = bernoulli_rng(p_jump);
      m = ou_mean[1] + jump_draw * mu_jump;
      s = sqrt(sd[1]^2 + jump_draw * sigma_jump^2);
    } else {
      m = ou_mean[2];
      s = sd[2];
    }
    // update the y_pred and y_expected vectors;
    // y_pred[2] = normal_rng(m, s);
    // y_pred_expected[2] = pi[1] * (p_jump*(ou_mean[1] + mu_jump) + (1-p_jump)*ou_mean[1]) + pi[2] * (ou_mean[2]);
    
    log_predict_prob = log(pi);

    log_component_density[1] = log_predict_prob[1] + log_sum_exp(
      log(p_jump) + normal_lpdf( y[2] | ou_mean[1] + mu_jump, sqrt(sigma[1]^2/ (2*theta[1]) + sigma_jump^2)),
      log1m(p_jump) + normal_lpdf( y[2] | ou_mean[1] , sigma[1] / sqrt(2*theta[1]) )
    );
    log_component_density[2] = log_predict_prob[2] + normal_lpdf( y[2] | ou_mean[2], sigma[2] / sqrt(2*theta[2]));

    log_normalising_const = log_sum_exp(log_component_density);
    log_filter_prob[1] =  log_component_density[1] - log_normalising_const;
    log_filter_prob[2] = log_component_density[2] - log_normalising_const;

    for (t in 2:(T-1)) {
      // latent transition densities, alpha_{t|t-1}
      log_predict_prob[1] = log_sum_exp(
        log_A[1,1] + log_filter_prob[1], 
        log_A[2,1] + log_filter_prob[2]
        );
      log_predict_prob[2] = log_sum_exp(
        log_A[1,2] + log_filter_prob[1], 
        log_A[2,2] + log_filter_prob[2]
        );

      // Randomly draw the regime of the time-point using the predicted probability
      // \alpha^k_{t|t-1} = Pr(Z_i=k | y_{1:t-1})
      int pred_regime = categorical_rng(exp(log_predict_prob));

      ou_mean[1] = deterministic_mean[t+1] + alpha[1] * (y[t] - deterministic_mean[t]); 
      ou_mean[2] = deterministic_mean[t+1] + alpha[2] * (y[t] - deterministic_mean[t]); 

      // Define the mean, m, and standard deviation, s, for the random draw
      if (pred_regime==1) {
        int jump_draw = bernoulli_rng(p_jump);
        m = ou_mean[1] + jump_draw * mu_jump;
        s = sqrt(sd[1]^2 + jump_draw * sigma_jump^2);
      } else {
        m = ou_mean[2];
        s = sd[2];
      }
      // y_pred[t+1] = normal_rng(m, s);
      // y_pred_expected[t+1] = exp(log_predict_prob[1]) * (p_jump*(ou_mean[1] + mu_jump) + (1-p_jump)*ou_mean[1]) + exp(log_predict_prob[2]) * (ou_mean[2]);

      // Calculate the obs transition densities, f(y_t | w_{1:(t-1)})
      log_emit[1] = log_sum_exp(
        log(p_jump) + normal_lpdf(y[t+1] | ou_mean[1] + mu_jump, sqrt(sd[1]^2 + sigma_jump^2)),
        log1m(p_jump) + normal_lpdf(y[t+1] | ou_mean[1], sd[1])
      );
      log_emit[2] = normal_lpdf(y[t+1] | ou_mean[2], sd[2]);
      
      // log-likelihood
      log_component_density[1] = log_predict_prob[1] + log_emit[1];
      log_component_density[2] = log_predict_prob[2] + log_emit[2];

      // latent probabilities, alpha_t
      log_normalising_const = log_sum_exp(log_component_density);
      log_filter_prob[1] = log_predict_prob[1] + log_emit[1] - log_normalising_const;
      log_filter_prob[2] = log_predict_prob[2] + log_emit[2] - log_normalising_const;
    }
    final_log_predict_prob = log_predict_prob;
  }

  // -- Out-of-Sample forecast --------------------------------------------------
  // E[y[t] | y[1:T]], t>T obtained by marginalising over z[t].
  // Uses *prior* state probabilities (predicted, not filtered) so it is a
  // genuine out-of-sample predictive quantity.
  vector[T_fore] y_forecast;
  vector[T_fore] y_forecast_expected;
  // t = 1 has no predecessor, so set to the observed value
  {
    // -- transformed parameters

    array[T_fore] int state_forecast;
    vector[2] ou_mean;
    real m;
    real s;

    // Draw initial state for the final observation at time T
    // final_log_predict_prob: is saved from the one-step-ahead predicition
    int initial_state = categorical_rng(softmax(final_log_predict_prob));
    
    // Draw the first out-of-sample state, at time T+1, conditioning on `initial_state`
    state_forecast[1] = categorical_rng(A[initial_state]);
    // calculate the mean of the OU processes, conditioning on y_{1:T}
    ou_mean[1] = deterministic_mean_fore[1] + alpha[1] * (y[T] - deterministic_mean[T]); 
    ou_mean[2] = deterministic_mean_fore[1] + alpha[2] * (y[T] - deterministic_mean[T]); 

    // Define the mean, m, and standard deviation, s, for the random draw
    if (state_forecast[1]==1) {
      int jump_draw = bernoulli_rng(p_jump);
      m = ou_mean[1] + jump_draw * mu_jump;
      s = sqrt(sd[1]^2 + jump_draw * sigma_jump^2);
    } else {
      m = ou_mean[2];
      s = sd[2];
    }
    y_forecast[1] = normal_rng(m, s);
    y_forecast_expected[1] = A[state_forecast[1], 1] * (p_jump*(ou_mean[1] + mu_jump) + (1-p_jump)*ou_mean[1]) + A[state_forecast[1], 2] * (ou_mean[2]);

    // Draw the next 2:N_fore forecasts. 
    for (t in 2:T_fore) {
       // Draw the state, condition on state_forecast[t-1]
      state_forecast[t] = categorical_rng(A[state_forecast[1]]);

      ou_mean[1] = deterministic_mean_fore[t] + alpha[1] * (y_forecast[t-1] - deterministic_mean_fore[t-1]); 
      ou_mean[2] = deterministic_mean_fore[t] + alpha[2] * (y_forecast[t-1] - deterministic_mean_fore[t-1]); 

      // Define the mean, m, and standard deviation, s, for the random draw
      if (state_forecast[t]==1) {
        int jump_draw = bernoulli_rng(p_jump);
        m = ou_mean[1] + jump_draw * mu_jump;
        s = sqrt(sd[1]^2 + jump_draw * sigma_jump^2);
      } else {
        m = ou_mean[2];
        s = sd[2];
      }
      y_forecast[t] = normal_rng(m, s);
      y_forecast_expected[t] = A[state_forecast[t], 1] * (p_jump*(ou_mean[1]+mu_jump) + (1-p_jump)*ou_mean[1]) + A[state_forecast[t],2] * ou_mean[2];
    }
  }
}
*/



