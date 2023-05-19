import sampling

if __name__=="__main__":
	sampler = sampling.Sampler(n_query=2, dim_features=3, update_func="approx",beta_demo=0.1, beta_pref=5.)
	s = sampler.sample(50000)

