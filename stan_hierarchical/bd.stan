#include FUNCTIONS-DATA.stan
transformed data {
  vector<lower=0>[I] delta = rep_vector(1, I);
#include DATA-COUNTS.stan
}
parameters {
  real<lower=0, upper=1> pi;
  real mu_alpha;
  real<lower=0> tau_alpha;
  vector[J] z_alpha;
  vector[I] beta;
  vector<lower=0, upper=1>[I] lambda;
}
transformed parameters {
  real log_lik;
  vector[J] alpha_acc;
  vector[J] logit_alpha_sens;
  vector[J] logit_alpha_spec;

  alpha_acc = mu_alpha + tau_alpha * z_alpha;
  logit_alpha_sens = alpha_acc;
  logit_alpha_spec = alpha_acc;

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
  pi ~ beta(2, 2);
  mu_alpha ~ normal(1, 2);
  tau_alpha ~ lognormal(0, 1);
  z_alpha ~ std_normal();
  beta ~ normal(0, 1);
  lambda ~ beta(2, 2);
  target += log_lik;
}
#include GQ.stan