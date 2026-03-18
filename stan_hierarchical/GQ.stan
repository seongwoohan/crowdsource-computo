generated quantities {
  array[I] real log_lik_pointwise;

  array[J + 1] int<lower=0> votes_sim;
  array[J + 1] int<lower=0, upper=1> votes_sim_lt_data;

  array[J] int<lower=0> rater_sim = rep_array(0, J);
  array[J] int<lower=0, upper=1> rater_sim_lt_data;

  {
    vector[I] lp_pos = rep_vector(log(pi), I);
    vector[I] lp_neg = rep_vector(log1m(pi), I);

    array[N] int rating_sim;
    array[I] int z_sim;

    for (n in 1:N) {
      int i = item[n];
      int j = rater[n];
      int y = rating[n];

      lp_pos[i] += bernoulli_lpmf(
        y | lambda[i] + (1 - lambda[i]) * inv_logit(delta[i] * (logit_alpha_sens[j] - beta[i]))
      );
      lp_neg[i] += bernoulli_lpmf(
        y | (1 - lambda[i]) * inv_logit(-delta[i] * (logit_alpha_spec[j] - beta[i]))
      );
    }

    for (i in 1:I) {
      log_lik_pointwise[i] = log_sum_exp(lp_pos[i], lp_neg[i]);
      z_sim[i] = bernoulli_rng(pi);
    }

    for (n in 1:N) {
      int i = item[n];
      int j = rater[n];
      rating_sim[n] =
        bernoulli_rng(
          z_sim[i] == 1
            ? lambda[i] + (1 - lambda[i]) * inv_logit(delta[i] * (logit_alpha_sens[j] - beta[i]))
            : (1 - lambda[i]) * inv_logit(-delta[i] * (logit_alpha_spec[j] - beta[i]))
        );
    }

    votes_sim = vote_count(rating_sim, item, rater, I, J);
    votes_sim_lt_data = lte_sims_rng(votes_sim, votes_data);

    rater_sim = rater_count(rating_sim, rater, J);
    rater_sim_lt_data = lte_sims_rng(rater_sim, rater_data);
  }
}