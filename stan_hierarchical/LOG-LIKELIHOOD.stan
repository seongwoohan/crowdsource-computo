  real log_lik;

  {
    vector[I] log_liks;
    vector[I] lp_pos = rep_vector(log(pi), I);
    vector[I] lp_neg = rep_vector(log1m(pi), I);
    for (n in 1:N) {
      int i = item[n];
      int j = rater[n];
      int y = rating[n];
      lp_pos[i] += bernoulli_lpmf(y | lambda[i] + (1 - lambda[i]) * inv_logit(delta[i] * (alpha_sens[j] - beta[i])));
      lp_neg[i] += bernoulli_lpmf(y | (1 - lambda[i]) * inv_logit(-delta[i] * (alpha_spec[j] - beta[i])));
    }
    for (i in 1:I) {
      log_liks[i] = log_sum_exp(lp_pos[i], lp_neg[i]);
    }
    log_lik = sum(log_liks);
  }
