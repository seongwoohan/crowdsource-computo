#include FUNCTIONS-DATA.stan
transformed data {
  vector[I] beta = rep_vector(0, I);
  vector<lower=0>[I] delta = rep_vector(1, I);
  vector<lower=0, upper=1>[I] lambda = rep_vector(0, I);
#include DATA-COUNTS.stan
}
parameters {
  real<lower=0, upper=1> pi;
  real alpha_spec_scalar;
  real<lower=-alpha_spec_scalar> alpha_sens_scalar;
}
transformed parameters {
  real log_lik;
  vector[J] logit_alpha_spec = rep_vector(alpha_spec_scalar, J);
  vector[J] logit_alpha_sens = rep_vector(alpha_sens_scalar, J);

  {
    vector[I] log_liks;
    vector[I] lp_pos = rep_vector(log(pi), I);
    vector[I] lp_neg = rep_vector(log1m(pi), I);

    for (n in 1:N) {
      int i = item[n];
      int j = rater[n];
      int y = rating[n];

      lp_pos[i] += bernoulli_lpmf(y | inv_logit(logit_alpha_sens[j]));
      lp_neg[i] += bernoulli_lpmf(y | inv_logit(-logit_alpha_spec[j]));
    }

    for (i in 1:I) {
      log_liks[i] = log_sum_exp(lp_pos[i], lp_neg[i]);
    }
    log_lik = sum(log_liks);
  }
}
model {
  pi ~ beta(2, 2);
  logit_alpha_spec ~ normal(2, 2);
  logit_alpha_sens ~ normal(1, 2);
  target += log_lik;
}
#include GQ.stan