generated quantities {
  array[N] real log_lik_pointwise;  // Array to store pointwise log-likelihoods

  for (n in 1:N) {
    int i = item[n];
    int j = rater[n];
    
    // Compute the pointwise log-likelihood for each observation
    log_lik_pointwise[n] = log_mix(pi,
      bernoulli_lpmf(rating[n] | lambda[i] + (1 - lambda[i]) * inv_logit(delta[i] * (alpha_sens[j] - beta[i]))),
      bernoulli_lpmf(rating[n] | (1 - lambda[i]) * inv_logit(-delta[i] * (alpha_spec[j] - beta[i]))));
  }
}

