#include "decs.h"
#include "compton.h"
#include "model_radiation.h"

static const double K0_MAX = 1.e6;
static const double GAMMA_E_MIN = 1.0;
static const double BETA_E_MIN = 1.e-12;
static const int SAMPLE_KLEIN_NISHINA_MAX_ATTEMPTS = 10000000;
static const int SAMPLE_ELECTRON_MAX_ATTEMPTS = 10000000;
static const int SAMPLE_BETA_DIST_MAX_ATTEMPTS = 10000000;

// Finding D6 (docs/2026-07-23_jet_implementation_changes.md sec 13-14): the legacy
// flux-factor rejection sampler in sample_electron_distr_p() has acceptance
// ~P(K<=K0_MAX)*sigma_KN(K)/sigma_T, which collapses to ~5e-8 in the regime the
// post-H1-fix jet supplement creates (Thetae at the THETAE_HARD_MAX=1e3 cap against
// multiply-scattered photons, k0 ~ 1e3) -- observed exhausting all 10M attempts and
// killing the run (SLURM job 778136). The analytic deep-KN sampler below is used for
// Thetae >= this threshold (chosen 2026-07-26, user decision on D6) and as a
// last-resort fallback when the legacy loop exhausts its attempt budget.
static const double DEEP_KN_SAMPLER_THETAE_MIN = 100.;
// Truncation of the Gamma(2,Thetae) proposal; Maxwell-Juttner tail mass above
// 30*Thetae is ~3e-12, negligible against Monte Carlo statistics.
static const double DEEP_KN_GAMMA_CAP_FACTOR = 30.;

static void fail_sampling(const char *detail)
{
	SET_RUN_STATUS(RUN_STATUS_SAMPLING_ERROR, "sampling_error", detail);
	fprintf(stderr, "sampling_error: %s\n", detail);
	fflush(stderr);
	exit(-1);
}

/*

Routines for treating Compton scattering via Monte Carlo.

Sampling procedures for electron distribution is based on
Canfield, Howard, and Liang, 1987, ApJ 323, 565.

*/

/*
   given photon w/ wavevector $k$ colliding w/ electron with
   momentum $p$, ($p$ is actually the four-velocity) 
   find new wavevector $kp$ 
   
*/

void sample_scattered_photon(double k[4], double p[4], double kp[4])
{
	double ke[4], kpe[4];
	double k0p;
	double n0x, n0y, n0z, n0dotv0, v0x, v0y, v0z, v1x, v1y, v1z, v2x,
	    v2y, v2z, v1, dir1, dir2, dir3;
	double cth, sth, phi, cphi, sphi;

	// boost into the electron frame
	// ke == photon momentum in elecron frame

	boost(k, p, ke);
	if (!(ke[0] > 0.0) || !isfinite(ke[0])) {
		for (int n = 0; n < 4; n++)
			kp[n] = k[n];
		return;
	}
	if (ke[0] > K0_MAX) {
		double scale = K0_MAX / ke[0];
		for (int n = 0; n < 4; n++)
			ke[n] *= scale;
	}
	if (ke[0] > 1.e-4) {
		k0p = sample_klein_nishina(ke[0]);
		cth = 1. - 1 / k0p + 1. / ke[0];
	} else {
		k0p = ke[0];
		cth = sample_thomson();
	}
	sth = sqrt(fabs(1. - cth * cth));

	// unit vector 1 for scattering coordinate system is
	// oriented along initial photon wavevector 
	//v0x = ke[1] / ke[0];
	//v0y = ke[2] / ke[0];
	//v0z = ke[3] / ke[0];

	// Explicitly compute kemag instead of using ke[0] to ensure that photon
  // is created normalized and doesn't inherit light cone errors from the
  // original superphoton
  double kemag = sqrt(ke[1]*ke[1] + ke[2]*ke[2] + ke[3]*ke[3]);
  if (!(kemag > 0.0) || !isfinite(kemag)) {
    for (int n = 0; n < 4; n++)
      kp[n] = k[n];
    return;
  }
  v0x = ke[1]/kemag;
  v0y = ke[2]/kemag;
  v0z = ke[3]/kemag;

	// randomly pick zero-angle for scattering coordinate system.
	// There's undoubtedly a better way to do this.
	monty_ran_dir_3d(&n0x, &n0y, &n0z);
	n0dotv0 = v0x * n0x + v0y * n0y + v0z * n0z;

	// unit vector 2
	v1x = n0x - (n0dotv0) * v0x;
	v1y = n0y - (n0dotv0) * v0y;
	v1z = n0z - (n0dotv0) * v0z;
	v1 = sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
	v1x /= v1;
	v1y /= v1;
	v1z /= v1;

	// find one more unit vector using cross product;
	// this guy is automatically normalized
	v2x = v0y * v1z - v0z * v1y;
	v2y = v0z * v1x - v0x * v1z;
	v2z = v0x * v1y - v0y * v1x;

	// now resolve new momentum vector along unit vectors 
	// create a four-vector $p$
	// solve for orientation of scattered photon 

	// find phi for new photon 
	phi = 2. * M_PI * monty_rand();
  sphi = sin(phi);
  cphi = cos(phi);

	p[1] *= -1.;
	p[2] *= -1.;
	p[3] *= -1.;

	dir1 = cth * v0x + sth * (cphi * v1x + sphi * v2x);
	dir2 = cth * v0y + sth * (cphi * v1y + sphi * v2y);
	dir3 = cth * v0z + sth * (cphi * v1z + sphi * v2z);

	kpe[0] = k0p;
	kpe[1] = k0p * dir1;
	kpe[2] = k0p * dir2;
	kpe[3] = k0p * dir3;

	// transform k back to lab frame
	boost(kpe, p, kp);

	// quality control
	if (!(kp[0] > 0.0) || !isfinite(kp[0])) {
		for (int n = 0; n < 4; n++)
			kp[n] = k[n];
		return;
	}
	if (kp[0] > K0_MAX) {
		double scale = K0_MAX / kp[0];
		for (int n = 0; n < 4; n++)
			kp[n] *= scale;
	}

	// done!
}

/*

Lorentz boost vector v into frame given by four-velocity u.
Result goes out in vp.
Assumes all four-velocities are given in orthonormal coordinates.

*/

void boost(double v[4], double u[4], double vp[4])
{
	double g, V, n1, n2, n3, gm1;

	g = u[0];
	V = sqrt(fabs(1. - 1. / (g * g)));
	n1 = u[1] / (g * V + SMALL);
	n2 = u[2] / (g * V + SMALL);
	n3 = u[3] / (g * V + SMALL);
	gm1 = g - 1.;

	// general Lorentz boost into frame u from lab frame 
	vp[0] = u[0]*v[0] - 
		u[1]*v[1] - 
		u[2]*v[2] - 
		u[3]*v[3];
	vp[1] = -u[1] * v[0] + 
		(1. + n1 * n1 * gm1) * v[1] +
	    	n1 * n2 * gm1 * v[2] + 
		n1 * n3 * gm1 * v[3];
	vp[2] = -u[2] * v[0] + 
		n2 * n1 * gm1 * v[1] + 
		(1. + n2 * n2 * gm1) * v[2] +
	    	n2 * n3 * gm1 * v[3];
	vp[3] = -u[3] * v[0] + 
		n3 * n1 * gm1 * v[1] + 
		n3 * n2 * gm1 * v[2] +
	    	(1. + n3 * n3 * gm1) * v[3];

}

/* return a cos(theta) consistent w/ Thomson
   differential cross section */

/* uses simple rejection scheme */

double sample_thomson()
{
	double x1, x2;

	do {

		x1 = 2. * monty_rand() - 1.;
		x2 = (3. / 4.) * monty_rand();

	} while (x2 >= (3. / 8.) * (1. + x1 * x1));

	return (x1);
}

/*

sample Klein-Nishina differential cross section.

This routine is inefficient; needs improvement.

*/

double sample_klein_nishina(double k0)
{
	double k0pmin, k0pmax, k0p_tent, x1;
	int n = 0;

	// Finding D6: composition-rejection sampler (Butcher & Messel 1960, the
	// standard EGS/Geant technique) for the deep-KN regime. The flat-box rejection
	// below has limiting efficiency log(2 k0)/(2 k0) (its own comment) -- ~1e5
	// wasted draws per event at the K0_MAX-clamped energies the post-H1-fix jet
	// supplement produces. This branch is O(1)-efficient (~40%) at any k0 and
	// samples the IDENTICAL target: with eps = k0p/k0 and ch = 1 + 1/k0 - 1/k0p,
	//     (1/eps + eps) * (1 - eps*sin2th/(1+eps^2))  ==  k0/k0p + k0p/k0 - 1 + ch^2,
	// i.e. exactly klein_nishina(k0, k0p) up to its constant 1/k0^2 factor.
	// Gated to k0 >= 1 so every previously-validated low-energy path is untouched
	// (at k0 < 1 the legacy box sampler's efficiency is >~ 35%). Prototype
	// validation (vs legacy at k0 = 0.5/1/10, |z| <= 1.4 at N=200k) recorded in
	// docs/2026-07-23_jet_implementation_changes.md sec 14.
	if (k0 >= 1.) {
		double eps0 = 1. / (1. + 2. * k0);
		double alph1 = log(1. / eps0);
		double alph2 = 0.5 * (1. - eps0 * eps0);
		double eps, t, s2;
		do {
			if (monty_rand() * (alph1 + alph2) < alph1) {
				eps = eps0 * exp(alph1 * monty_rand()); // density prop. to 1/eps
			} else {
				eps = sqrt(eps0 * eps0 + (1. - eps0 * eps0) * monty_rand()); // prop. to eps
			}
			t = (1. - eps) / (k0 * eps); // = 1 - cos(theta) from the Compton relation
			s2 = t * (2. - t);           // = sin^2(theta)
			n++;
			if (n > SAMPLE_KLEIN_NISHINA_MAX_ATTEMPTS) {
				char detail[RUN_STATUS_DETAIL_MAXLEN];
				snprintf(detail, RUN_STATUS_DETAIL_MAXLEN,
				         "sample_klein_nishina_bm_stalled k0=%g attempts=%d",
				         k0, n);
				fail_sampling(detail);
			}
		} while (monty_rand() >= 1. - eps * s2 / (1. + eps * eps));
		return eps * k0;
	}

	// a low efficiency sampling algorithm, particularly for large k0;
	// limiting efficiency is log(2 k0)/(2 k0)
	k0pmin = k0 / (1. + 2. * k0);	 // at theta = Pi
	k0pmax = k0;  // at theta = 0
	do {

		// tentative value
		k0p_tent = k0pmin + (k0pmax - k0pmin) * monty_rand();

		// rejection sample in box of height = kn(kmin)
		x1 = 2. * (1. + 2. * k0 +
			   2. * k0 * k0) / (k0 * k0 * (1. + 2. * k0));
		x1 *= monty_rand();

		n++;
		if (n > SAMPLE_KLEIN_NISHINA_MAX_ATTEMPTS) {
			char detail[RUN_STATUS_DETAIL_MAXLEN];
			snprintf(detail, RUN_STATUS_DETAIL_MAXLEN,
			         "sample_klein_nishina_stalled k0=%g attempts=%d",
			         k0, n);
			fail_sampling(detail);
		}

	} while (x1 >= klein_nishina(k0, k0p_tent));

	return (k0p_tent);
}

/*  

   differential cross section for scattering from 
   frequency a -> frequency ap.  Frequencies are
   in units of m_e.  Unnormalized!
   
*/

double klein_nishina(double a, double ap)
{
	double ch, kn;

	ch = 1. + 1. / a - 1. / ap;
	kn = (a / ap + ap / a - 1. + ch * ch) / (a * a);

	return (kn);
}

/*

	sample electron distribution to find which electron was
	scattered.

*/

// Klein-Nishina total cross section / Thomson. Exactly the expression that lived
// inline in sample_electron_distr_p's rejection loop; factored out so the deep-KN
// sampler and the legacy loop share one definition.
static double sigma_kn_over_thomson(double K)
{
	if (K < 1.e-3)
		return 1. - 2. * K;
	return (3. / (4. * K * K)) *
	       (2. + K * K * (1. + K) / ((1. + 2. * K) * (1. + 2. * K)) +
	        (K * K - 2. * K - 2.) / (2. * K) * log(1. + 2. * K));
}

// Finding D6: analytic deep-KN electron sampler (Maxwell-Juttner only).
//
// Samples (gamma_e, mu) from the SAME target the legacy loop samples:
//     MJ(gamma) * (1 - beta*mu)/2 * sigma_KN(K)/sigma_T,   K = gamma*(1-beta*mu)*k0,
// truncated to K <= K0_MAX (identical to the legacy loop's `K > K0_MAX -> continue`).
//
// Change of proposal: gamma ~ Gamma(2, Thetae) (density prop. to gamma*e^{-gamma/Th},
// sampled exactly as -Thetae*log(u1*u2)) and mu ~ Uniform(-1,1). Writing the flux
// factor as (1 - beta*mu) = K/(gamma*k0), the exact acceptance ratio collapses to
//     A = beta * [K * sigma_KN(K)/sigma_T] / G_sup
// -- the flux factor cancels against the 1/K falloff of the deep-KN cross section,
// which is precisely why the legacy proposal (which FAVORS head-on, high-K draws)
// dies in this regime. g(K) = K*sigma_KN(K)/sigma_T is monotone increasing (verified
// numerically over K in [1e-10, 1e8]), so G_sup = 1.02 * g(K_cap) with
// K_cap = min(2*gamma_cap*k0, K0_MAX) majorizes it with a 2% safety margin.
//
// Efficiency is O(1) at any K: ~10% at the exact regime that exhausted the legacy
// sampler (Thetae=1e3, k0~2e3, job 778136) where legacy acceptance was ~5e-8, and
// a few percent in the Thomson limit (where it exactly reduces to the flux-factor
// target). Prototype validation (distributional equivalence vs legacy at Thetae=100
// across k0 = 1e-8 / 0.1 / 3.0, all |z| < 2.6 at N=150k) recorded in
// docs/2026-07-23_jet_implementation_changes.md sec 14.
static void sample_electron_mu_deep_kn(double k0_local, double Thetae,
                                       double *gamma_out, double *beta_out,
                                       double *mu_out)
{
	double gamma_cap = DEEP_KN_GAMMA_CAP_FACTOR * Thetae;
	double K_cap = 2. * gamma_cap * k0_local;
	if (K_cap > K0_MAX)
		K_cap = K0_MAX;
	double G_sup = 1.02 * K_cap * sigma_kn_over_thomson(K_cap);
	int attempts = 0;

	while (1) {
		if (attempts > SAMPLE_ELECTRON_MAX_ATTEMPTS) {
			char detail[RUN_STATUS_DETAIL_MAXLEN];
			snprintf(detail, RUN_STATUS_DETAIL_MAXLEN,
			         "sample_electron_deep_kn_stalled Thetae=%g k0=%g attempts=%d",
			         Thetae, k0_local, attempts);
			fail_sampling(detail);
		}
		attempts++;

		double u1 = monty_rand();
		double u2 = monty_rand();
		if (!(u1 > 0.) || !(u2 > 0.))
			continue;
		double ge = -Thetae * log(u1 * u2); // Gamma(2, Thetae)
		if (ge <= 1. || ge >= gamma_cap)
			continue;
		double be = sqrt(1. - 1. / (ge * ge));
		double mu = 2. * monty_rand() - 1.;
		double K = ge * (1. - be * mu) * k0_local;
		if (!(K > 0.0) || !isfinite(K) || K > K0_MAX)
			continue;
		if (monty_rand() * G_sup < be * K * sigma_kn_over_thomson(K)) {
			*gamma_out = ge;
			*beta_out = be;
			*mu_out = mu;
			return;
		}
	}
}

void sample_electron_distr_p(double k[4], double p[4], double Thetae)
{
	double beta_e, mu = 0., phi, cphi, sphi, gamma_e = 0., sigma_KN = 0.;
	double K = 0., sth, cth, x1 = 0., n0dotv0, v0, v1;
	double n0x, n0y, n0z;
	double v0x, v0y, v0z;
	double v1x, v1y, v1z;
	double v2x, v2y, v2z;
	int sample_cnt = 0;
	double k0_local = (k[0] > 0.0 && isfinite(k[0])) ? k[0] : fabs(k[0]);
	if (!(k0_local > 0.0) || !isfinite(k0_local)) {
		k0_local = 1.e-30;
	}

	// Finding D6: route hot zones to the analytic deep-KN sampler. Threshold per
	// the 2026-07-26 decision; Maxwell-Juttner builds only (the analytic proposal
	// is MJ-specific -- kappa/power-law builds keep the legacy path unchanged).
	int use_deep_kn = 0;
#if MODEL_EDF == EDF_MAXWELL_JUTTNER
	if (Thetae >= DEEP_KN_SAMPLER_THETAE_MIN)
		use_deep_kn = 1;
#endif

	if (use_deep_kn) {
		sample_electron_mu_deep_kn(k0_local, Thetae, &gamma_e, &beta_e, &mu);
	} else {
	while (1) {
		if (sample_cnt > SAMPLE_ELECTRON_MAX_ATTEMPTS) {
#if MODEL_EDF == EDF_MAXWELL_JUTTNER
			// Finding D6: this used to fail_sampling() -> exit(-1), killing the
			// entire run because one photon was un-sampleable. Fall back to the
			// analytic sampler instead -- it is efficient precisely in the regime
			// that exhausts this loop.
			fprintf(stderr,
			        "sample_electron: legacy sampler exhausted %d attempts "
			        "(Thetae=%g k0=%g); using deep-KN analytic sampler\n",
			        sample_cnt, Thetae, k0_local);
			sample_electron_mu_deep_kn(k0_local, Thetae, &gamma_e, &beta_e, &mu);
			break;
#else
			char detail[RUN_STATUS_DETAIL_MAXLEN];
			snprintf(detail, RUN_STATUS_DETAIL_MAXLEN,
			         "sample_electron_stalled Thetae=%g mu=%g gamma_e=%g K=%g sigma_KN=%g x1=%g attempts=%d",
			         Thetae, mu, gamma_e, K, sigma_KN, x1, sample_cnt);
			fail_sampling(detail);
#endif
		}

		sample_cnt++;
		sample_beta_distr(Thetae, &gamma_e, &beta_e);
		if (!(gamma_e > GAMMA_E_MIN) || !isfinite(gamma_e)) {
			gamma_e = GAMMA_E_MIN;
		}
		if (!isfinite(beta_e) || beta_e < 0.0) {
			beta_e = sqrt(1. - 1. / (gamma_e * gamma_e));
		}
		if (beta_e < BETA_E_MIN)
			beta_e = BETA_E_MIN;

		mu = sample_mu_distr(beta_e);
		// sometimes |mu| > 1 from roundoff error, fix it
		if (mu > 1.)
			mu = 1.;
		else if (mu < -1.)
			mu = -1;

		// frequency in electron rest frame
		K = gamma_e * (1. - beta_e * mu) * k0_local;
		if (!(K > 0.0) || !isfinite(K) || K > K0_MAX) {
			continue;
		}

		sigma_KN = sigma_kn_over_thomson(K);

		x1 = monty_rand();

		if (x1 < sigma_KN)
			break;
	}
	}

	// first unit vector for coordinate system 
	v0x = k[1];
	v0y = k[2];
	v0z = k[3];
	v0 = sqrt(v0x * v0x + v0y * v0y + v0z * v0z);
	v0x /= v0;
	v0y /= v0;
	v0z /= v0;

	// pick zero-angle for coordinate system 
	monty_ran_dir_3d(&n0x, &n0y, &n0z);
	n0dotv0 = v0x * n0x + v0y * n0y + v0z * n0z;

	// second unit vector
	v1x = n0x - (n0dotv0) * v0x;
	v1y = n0y - (n0dotv0) * v0y;
	v1z = n0z - (n0dotv0) * v0z;

	// normalize
	v1 = sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
	v1x /= v1;
	v1y /= v1;
	v1z /= v1;

	// find one more unit vector using cross product;
	// this guy is automatically normalized
	v2x = v0y * v1z - v0z * v1y;
	v2y = v0z * v1x - v0x * v1z;
	v2z = v0x * v1y - v0y * v1x;

	// now resolve new momentum vector along unit vectors 
	// and create a four-vector $p$
	phi = monty_rand() * 2. * M_PI;	// orient uniformly
  sphi = sin(phi);
  cphi = cos(phi);
	cth = mu;
	sth = sqrt(1. - mu * mu);

	p[0] = gamma_e;
	p[1] = gamma_e * beta_e * (cth * v0x + sth * (cphi * v1x + sphi * v2x));
	p[2] = gamma_e * beta_e * (cth * v0y + sth * (cphi * v1y + sphi * v2y));
	p[3] = gamma_e * beta_e * (cth * v0z + sth * (cphi * v1z + sphi * v2z));

	if (beta_e < 0) {
		fprintf(stderr, "betae error: %g %g %g %g\n",
			p[0], p[1], p[2], p[3]);
	}

	return;
}

/* 
   sample dimensionless speed of electron
   from relativistic maxwellian 

   checked. 
   
*/

// Function that, when zero, gives gamma for which dN/d log gam is maximized
double dfdgam(double ge, void *params)
{
  double Thetae = *(double *)params;

#if MODEL_EDF==EDF_KAPPA_FIXED
  double kap = model_kappa;
  double w = kappa_w(Thetae, kap);
  return 2. + ge * ( ge / ( ge*ge - 1 ) - 1. / GAMMACUT - (kap+1)/kap/w / (1. + (ge-1.)/kap/w) );
#elif MODEL_EDF==EDF_MAXWELL_JUTTNER
  return ge - pow(ge,3.) - 2.*Thetae + 3.*pow(ge,2.)*Thetae;
#elif MODEL_EDF==EDF_POWER_LAW
  (void)Thetae;
  fprintf(stderr, "power law EDF not supported with dfdgam\n");
  exit(3);
#else
  fprintf(stderr, "must choose valid MODEL_EDF\n");
  exit(3);
#endif
}

// electron distribution function (Maxwell-Juettner or kappa below)
// dN / d log gamma
double fdist(double ge, double Thetae)
{
#if MODEL_EDF==EDF_KAPPA_FIXED
  double kap = model_kappa;
  double w = kappa_w(Thetae, kap);
  return ge*ge*sqrt(ge*ge - 1.)*pow(1. + (ge - 1.)/(kap * w), - kap - 1.)*exp(-ge/GAMMACUT);
#elif MODEL_EDF==EDF_MAXWELL_JUTTNER
  return ge*ge*sqrt(ge*ge-1.)*exp(-ge/Thetae);
#elif MODEL_EDF==EDF_POWER_LAW
  (void)Thetae;
  (void)ge;
  fprintf(stderr, "power law EDF not supported with fdist\n");
  exit(3);
#else
  fprintf(stderr, "must choose valid MODEL_EDF\n");
  exit(3);
#endif
}


void sample_powerlaw_distr(double *gamma_e, double *beta_e)
{
  double p = powerlaw_p;
  double gmin = powerlaw_gamma_min;
  double gmax = powerlaw_gamma_max;

  double x = monty_rand();

  *gamma_e = pow( (1.-x) * pow(gmin, 1.-p) + x * pow(gmax, 1.-p), 1./(1.-p) );
  *beta_e = sqrt(1. - 1. / (*gamma_e * *gamma_e));
}

#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
void sample_beta_distr(double Thetae, double *gamma_e, double *beta_e)
{
#if MODEL_EDF==EDF_POWER_LAW
  sample_powerlaw_distr(gamma_e, beta_e);
#else
  sample_beta_distr_num(Thetae, gamma_e, beta_e);
#endif
}

/*
void sample_beta_distr_y(double Thetae, double *gamma_e, double *beta_e) 
{
  double sample_y_distr_kappa(double);
#if MODEL_EDF==EDF_KAPPA_FIXED
  double y = sample_y_distr_kappa(Thetae);
#elif MODEL_EDF==EDF_MAXWELL_JUTTNER
  double y = sample_y_distr(Thetae);
#elif MODEL_EDF==EDF_POWER_LAW
  double y = 0.;
  fprintf(stderr, "power law EDF not supported with sample_beta_distr_y\n");
  exit(3);
#else
  fprintf(stderr, "must choose valid MODEL_EDF\n");
  exit(3);
#endif
  *gamma_e = y * y * Thetae + 1.;
  *beta_e = sqrt(1. - 1. / (*gamma_e * *gamma_e));

  return;
}
 */

void sample_beta_distr_num(double Thetae, double *gamma_e, double *beta_e)
{
  // Relativistic kappa distribution does not like very small Thetae. Ugly kludge.
  if (Thetae < 0.01) {
    *gamma_e = 1.000001;
	  *beta_e = sqrt(1. - 1. / (*gamma_e * *gamma_e));
    return;
  }

  // Get maximum for window
  int status, iter = 0, max_iter = 100;
  const gsl_root_fsolver_type *T;
  gsl_root_fsolver *s;
  double ge_max = 1. + Thetae;
  double ge_lo = GSL_MAX(1.000001, 0.01*Thetae); // dfdgam -> +inf as ge -> 1+
  double ge_hi = GSL_MAX(100., 1000.*Thetae);

  //printf("Thetae = %e ge_lo = %e ge_hi = %e\n", Thetae, ge_lo, ge_hi);
  gsl_function F;

  //printf("%e %e\n", dfdgam(ge_lo, &params), dfdgam(ge_hi, &params));
  F.function = &dfdgam;
  F.params = &Thetae;
  T = gsl_root_fsolver_brent;
  s = gsl_root_fsolver_alloc(T);
  gsl_root_fsolver_set(s, &F, ge_lo, ge_hi);
  do {
    iter++;
    status = gsl_root_fsolver_iterate(s);
    ge_max = gsl_root_fsolver_root(s);
    ge_lo = gsl_root_fsolver_x_lower(s);
    ge_hi = gsl_root_fsolver_x_upper(s);
    status = gsl_root_test_interval(ge_lo, ge_hi, 0, 0.001);
  } while (status == GSL_CONTINUE && iter < max_iter);

  double f_max = fdist(ge_max, Thetae);
  gsl_root_fsolver_free(s);
  if (!(f_max > 0.0) || !isfinite(f_max)) {
    char detail[RUN_STATUS_DETAIL_MAXLEN];
    snprintf(detail, RUN_STATUS_DETAIL_MAXLEN,
             "sample_beta_distr_invalid_fmax Thetae=%g f_max=%g ge_max=%g",
             Thetae, f_max, ge_max);
    fail_sampling(detail);
  }
  //fprintf(stderr, "max is %g at %g for %g\n", f_max, ge_max, Thetae);
  
  // Sample electron gamma
  double ge_samp;
  int sample_attempts = 0;
  double lge_min = log(GSL_MAX(1., 0.01*Thetae));
  double lge_max = log(GSL_MAX(100., 1000.*Thetae));
  do {
    sample_attempts++;
    if (sample_attempts > SAMPLE_BETA_DIST_MAX_ATTEMPTS) {
      char detail[RUN_STATUS_DETAIL_MAXLEN];
      snprintf(detail, RUN_STATUS_DETAIL_MAXLEN,
               "sample_beta_distr_stalled Thetae=%g f_max=%g attempts=%d",
               Thetae, f_max, sample_attempts);
      fail_sampling(detail);
    }
    ge_samp = exp(lge_min + (lge_max - lge_min)*monty_rand()); 
  } while (fdist(ge_samp, Thetae)/f_max < monty_rand());

  *gamma_e = ge_samp;                                                
  *beta_e = sqrt(1. - 1. / (*gamma_e * *gamma_e));
}

/* 

   sample y, which is the temperature-normalized
   kinetic energy.
   Uses procedure outlined in Canfield et al. 1987,
   p. 572 et seq. 
   
*/

double sample_y_distr(double Thetae)
{

	double S_3, pi_3, pi_4, pi_5, pi_6, y, x1, x2, x, prob;
	double num, den;

	pi_3 = sqrt(M_PI) / 4.;
	pi_4 = sqrt(0.5 * Thetae) / 2.;
	pi_5 = 3. * sqrt(M_PI) * Thetae / 8.;
	pi_6 = Thetae * sqrt(0.5 * Thetae);

	S_3 = pi_3 + pi_4 + pi_5 + pi_6;

	pi_3 /= S_3;
	pi_4 /= S_3;
	pi_5 /= S_3;
	pi_6 /= S_3;

	do {
		x1 = monty_rand();

		if (x1 < pi_3) {
			x = monty_ran_chisq(3);
		} else if (x1 < pi_3 + pi_4) {
			x = monty_ran_chisq(4);
		} else if (x1 < pi_3 + pi_4 + pi_5) {
			x = monty_ran_chisq(5);
		} else {
			x = monty_ran_chisq(6);
		}

		// this translates between defn of distr in
		// Canfield et al. and standard chisq distr
		y = sqrt(x / 2);

		x2 = monty_rand();
		num = sqrt(1. + 0.5 * Thetae * y * y);
		den = (1. + y * sqrt(0.5 * Thetae));

		prob = num / den;

	} while (x2 >= prob);

	return (y);
}

double sample_mu_distr(double beta_e)
{
	double mu, x1, det;

	x1 = monty_rand();
	det = 1. + 2. * beta_e + beta_e * beta_e - 4. * beta_e * x1;
	if (det < 0.)
		fprintf(stderr, "det < 0  %g %g\n\n", beta_e, x1);
	mu = (1. - sqrt(det)) / beta_e;
	return (mu);
}
