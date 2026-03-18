  real log_lik;

  matrix[2, J] alpha_latent;
  vector[J] eta_alpha;
  vector[J] logit_alpha_spec;
  vector[J] logit_alpha_sens;

  alpha_latent = rep_matrix(mu_alpha, J) + diag_pre_multiply(tau_alpha, L_Omega_alpha) * z_alpha;

  for (j in 1:J) {
    eta_alpha[j] = alpha_latent[1, j];
    logit_alpha_spec[j] = alpha_latent[2, j];
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