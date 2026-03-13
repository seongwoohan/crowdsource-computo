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
                          * inv_logit(delta[i] * (alpha_sens[j] - beta[i]))
                        : (1 - lambda[i])
                          * inv_logit(-delta[i] * (alpha_spec[j] -  beta[i])));
                        
    }

    votes_sim = vote_count(rating_sim, item, rater, I, J);
    votes_sim_lt_data = lte_sims_rng(votes_sim, votes_data);
    
    rater_sim = rater_count(rating_sim, rater, J);
    rater_sim_lt_data = lte_sims_rng(rater_sim, rater_data);
  }   
}

