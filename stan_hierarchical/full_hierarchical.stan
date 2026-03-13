functions {
  array[] int vote_count(array[] int rating,
                         array[] int item,
                         array[] int rater,
                         int I, int J) {
    int N = size(rating);
    array[I] int count_by_item = rep_array(1, I);  // index 0:5 by 1:6
    for (n in 1:N) {
      count_by_item[item[n]] += rating[n];
    }
    array[J + 1] int count = rep_array(0, J + 1);
    for (i in 1:I) {
      count[count_by_item[i]] += 1;
    }
    return count;
  }

  array[] int rater_count(array[] int rating,
                          array[] int rater,
                          int J) {
    array[J] int count = rep_array(0, J);
    int N = size(rating);
    for (n in 1:N) {
      count[rater[n]] += rating[n];
    }
    return count;
  }
  int lte_sim_rng(int x, int y) {
    return x == y ? bernoulli_rng(0.5) : x < y;
  }
  array[] int lte_sims_rng(array[] int x, array[] int y) {
    int N = size(x);
    array[N] int ltes;
    for (n in 1:N) {
      ltes[n] = lte_sim_rng(x[n], y[n]);
    }
    return ltes;
  }
}
data {
  int<lower=0> I;
  int<lower=0> J;
  int<lower=0> N;
  array[N] int<lower=1, upper=I> item;
  array[N] int<lower=1, upper=J> rater;
  array[N] int<lower=0, upper=1> rating;
}


transformed data {
  array[J + 1] int votes_data = vote_count(rating, item, rater, I, J);
  array[J] int rater_data = rater_count(rating, rater, J);
}

parameters {
  real<lower=0, upper=1> pi;

  vector[2] mu_alpha;
  vector<lower=0>[2] tau_alpha;
  cholesky_factor_corr[2] L_Omega_alpha;
  matrix[2, J] z_alpha;

  vector[I] beta;
  vector<lower=0>[I] delta;
  vector<lower=0, upper=1>[I] lambda;
}

transformed parameters {
  real log_lik;

  matrix[2, J] alpha_latent;
  vector[J] eta_alpha;
  vector[J] logit_alpha_spec;
  vector[J] logit_alpha_sens;

  alpha_latent = rep_matrix(mu_alpha, J) + diag_pre_multiply(tau_alpha, L_Omega_alpha) * z_alpha;

  for (j in 1:J) {
    eta_alpha[j] = alpha_latent[1, j];
    logit_alpha_spec[j] = alpha_latent[2, j];
    // cooperative-rater restriction:
    // logit_alpha_sens[j] > -logit_alpha_spec[j]
    logit_alpha_sens[j] = -logit_alpha_spec[j] + log1p_exp(eta_alpha[j]);
  }

  {
    vector[I] log_liks;
    vector[I] lp_pos = rep_vector(log(pi), I);
    vector[I] lp_neg = rep_vector(log1m(pi), I);
    for (n in 1:N) {
      int i = item[n];
      int j = rater[n];
      int y = rating[n];
      lp_pos[i] += bernoulli_lpmf(y | lambda[i] + (1 - lambda[i]) * inv_logit(delta[i] * (logit_alpha_sens[j] - beta[i])));
      lp_neg[i] += bernoulli_lpmf(y | (1 - lambda[i]) * inv_logit(-delta[i] * (logit_alpha_spec[j] - beta[i])));
    }
    for (i in 1:I) {
      log_liks[i] = log_sum_exp(lp_pos[i], lp_neg[i]);
    }
    log_lik = sum(log_liks);
  }
}

model {
  pi ~ beta(2, 2);

  mu_alpha ~ normal(0, 5);
  tau_alpha ~ lognormal(0, 1);
  L_Omega_alpha ~ lkj_corr_cholesky(2);
  to_vector(z_alpha) ~ std_normal();

  beta ~ normal(0, 1);
  delta ~ lognormal(0, 0.25);
  lambda ~ beta(2, 2);

  target += log_lik;
}

// generated quantities {
//   array[N] real log_lik_pointwise;  // Array to store pointwise log-likelihoods

//   for (n in 1:N) {
//     int i = item[n];
//     int j = rater[n];
    
//     // Compute the pointwise log-likelihood for each observation
//     log_lik_pointwise[n] = log_mix(pi,
//       bernoulli_lpmf(rating[n] | lambda[i] + (1 - lambda[i]) * inv_logit(delta[i] * (logit_alpha_sens[j] - beta[i]))),
//       bernoulli_lpmf(rating[n] | (1 - lambda[i]) * inv_logit(-delta[i] * (logit_alpha_spec[j] - beta[i]))));
//   }
// }

generated quantities {
  array[J + 1] int<lower = 0> votes_sim;
  array[J + 1] int<lower=0, upper=1> votes_sim_lt_data;

  array[J] int<lower=0> rater_sim = rep_array(0, J);
  array[J] int<lower=0, upper=1> rater_sim_lt_data;
  {
    array[N] int rating_sim;
    array[I] int z_sim;
    for (i in 1:I) {
      z_sim[i] = bernoulli_rng(pi);
    }
    for (n in 1:N) {
      int i = item[n];
      int j = rater[n];
      rating_sim[n]
        = bernoulli_rng(z_sim[i] == 1
                        ? lambda[i]
                        + (1 - lambda[i])
                          * inv_logit(delta[i] * (logit_alpha_sens[j] - beta[i]))
                        : (1 - lambda[i])
                          * inv_logit(-delta[i] * (logit_alpha_spec[j] -  beta[i])));
                        
    }

    votes_sim = vote_count(rating_sim, item, rater, I, J);
    votes_sim_lt_data = lte_sims_rng(votes_sim, votes_data);
    
    rater_sim = rater_count(rating_sim, rater, J);
    rater_sim_lt_data = lte_sims_rng(rater_sim, rater_data);
  }   
}