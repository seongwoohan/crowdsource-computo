import cmdstanpy as csp
import numpy as np
import scipy as sp
import pandas as pd
import logging
import warnings

# warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.filterwarnings( "ignore", module = "plotnine\..*" )
csp.utils.get_logger().setLevel(logging.ERROR)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


def rating_csv_to_dict(file):
    df = pd.read_csv(file, comment = '#')
    rater = df['rater'].to_list()
    item = df['item'].to_list()
    rating = df['rating'].to_list()
    I = int(np.max(item))
    J = int(np.max(rater))
    N = int(len(rater))
    data = { 'I': I, 'J': J, 'N': N,
             'item': item, 'rater': rater, 'rating': rating }
    return data

def sample(stan_file, data, init = {}):
    model = csp.CmdStanModel(stan_file = stan_file)
    sample = model.sample(data = data, inits = init,
                          iter_warmup=1000, iter_sampling=1000,
                          chains = 2, parallel_chains = 2,
                          show_console = True, show_progress=True,
                          refresh = 100,
                          seed = 925845)
    return sample

def pathfind(stan_file, data, init = {}):
    model = csp.CmdStanModel(stan_file = stan_file)
    fit = model.pathfinder(data = data, inits = init, show_console=True, refresh = 5)
    return fit

def min_p_twosided(ps):
    return np.fmin(np.min(ps), 1 - np.max(ps)) / 2

def p_twosided(ps):
    return np.fmin(ps, 1 - ps) / 2

#data_file = 'caries.csv'
data_file = 'rte.csv'
data_path = '../data/' + data_file
data = rating_csv_to_dict(data_path)
# init = {
#     'pi': 0.2,
#     'alpha_acc_scalar': 2,
#     'alpha_sens_scalar': 1,
#     'alpha_spec_scalar': 2,
#     'alpha_acc': np.full(data['J'], 2),
#     'alpha_sens': np.full(data['J'], 1),
#     'alpha_spec': np.full(data['J'], 2),
#     'beta': np.full(data['I'], 0),
#     'delta': np.full(data['I'], 1),
#     'lambda': np.full(data['I'], 0.5)
# }  
# init = {
#     'pi': 0.2,
#     'mu_alpha': np.array([1.0, 2.0]),
#     'tau_alpha': np.array([0.5, 0.5]),
#     'L_Omega_alpha': np.eye(2),
#     'z_alpha': np.zeros((2, data['J'])),
#     'beta': np.full(data['I'], 0.0),
#     'delta': np.full(data['I'], 1.0),
#     'lambda': np.full(data['I'], 0.5)
# } 
init = {
    'pi': 0.2,
    'mu_alpha': np.array([1.0, 2.0]),
    'tau_alpha': np.array([0.3, 0.3]),
    'L_Omega_alpha': np.array([[1.0, 0.0],
                               [0.1, np.sqrt(1 - 0.1**2)]]),
    'z_alpha': np.zeros((2, data['J'])),
    'beta': np.zeros(data['I']),
    'delta': np.ones(data['I']),
    'lambda': np.full(data['I'], 0.2)
}     
J = data['J']
    
# rater_labels = [f"rater_sim[{i}]" for i in range(1, J)]
# rater_lt_labels = [f"rater_sim_lt_data[{i}]" for i in range(1, J)]
# votes_labels = [f"votes_sim[{i}]" for i in range(1, J + 1)]
# votes_lt_labels = [f"votes_sim_lt_data[{i}]" for i in range(1, J + 1)]
rater_labels = [f"rater_sim[{i}]" for i in range(1, J + 1)]
rater_lt_labels = [f"rater_sim_lt_data[{i}]" for i in range(1, J + 1)]
votes_labels = [f"votes_sim[{i}]" for i in range(1, J + 2)]
votes_lt_labels = [f"votes_sim_lt_data[{i}]" for i in range(1, J + 2)]

# models = ['a', 'ab', 'abc', 'abcd', 'abcde', 'abce', 'abd', 'abde', 'ac', 'acd', 'ad', 'bc', 'bcd', 'bd', 'c', 'cd', 'd', 'full']
models = ['full', 'full_hierarchical']

rows = []
for model in models:
    print(f"***** {model = }")
    draws = sample('../stan_hierarchical/' + model + '.stan', data, init)
    post_summary = draws.summary()
    post_rhat = post_summary['R_hat']
    post_means = post_summary['Mean']
    rhat_lp = post_rhat['lp__']
    rhat_max = np.max(post_rhat)
    pi = post_means['pi']
    log_lik = post_means['log_lik']
    rater_sim = post_means[rater_labels]
    rater_lt_sim = post_means[rater_lt_labels]
    votes_sim = post_means[votes_labels]
    votes_lt_sim = post_means[votes_lt_labels]

    min_raters_p = min_p_twosided(rater_lt_sim)
    min_votes_p = min_p_twosided(votes_lt_sim)

    raters_p = p_twosided(rater_lt_sim)
    votes_p = p_twosided(votes_lt_sim)
    
    row = {'model': [model], 'rhat_max': [rhat_max], 'rhat_lp': [rhat_lp],
               'min_raters_p': min_raters_p, 'min_votes_p': min_votes_p,
               'rater_p': raters_p, 'ratings_p': votes_p,
               'pi': [pi], 'log_lik': log_lik }
    # row.update(dict(zip(rater_labels, rater_sim)))
    row.update(dict(zip(rater_lt_labels, rater_lt_sim)))
    # row.update(dict(zip(votes_labels, votes_sim)))
    row.update(dict(zip(votes_lt_labels, votes_lt_sim)))
    print(row)
    rows.append(pd.DataFrame(row))

results_df = pd.concat(rows)
results_df.to_csv('results-hierarchical-' + data_file, index=False, sep=',', encoding='utf-8')
    



