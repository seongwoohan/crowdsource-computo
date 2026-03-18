#include FUNCTIONS-DATA.stan
transformed data {
#include DATA-COUNTS.stan
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
#include LOG-LIKELIHOOD.stan
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
#include GQ.stan