#include FUNCTIONS-DATA.stan
transformed data {
  vector<lower=0>[I] delta = rep_vector(1, I);
  vector<lower=0, upper=1>[I] lambda = rep_vector(0, I);
  vector[J] logit_alpha_sens = rep_vector(0, J);
  vector[J] logit_alpha_spec = rep_vector(0, J);
#include DATA-COUNTS.stan
}
parameters {
  vector[I] beta;
}
transformed parameters {
  real<lower=0, upper=1> pi = 1;
  real log_lik;

  {
    vector[I] log_liks;
    vector[I] lp_pos = rep_vector(log(pi), I);
    vector[I] lp_neg = rep_vector(log1m(pi), I);

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
      log_liks[i] = log_sum_exp(lp_pos[i], lp_neg[i]);
    }
    log_lik = sum(log_liks);
  }
}
model {
  beta ~ normal(0, 1);
  target += log_lik;
}
#include GQ.stan