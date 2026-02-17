/*

model-independent radiation-related utilities.

*/

#include "decs.h"
#include "model_radiation.h"
#include "par.h"
#include "coordinates.h"

// this file defines:
//
//   Bnu_inv
//   jnu_inv
//   alpha_inv_scatt
//   alpha_inv_abs
//   kappa_es
//   get_fluid_nu
//   get_bk_angle
//

double model_kappa = 4.;

double powerlaw_gamma_cut = 1.e10;
double powerlaw_gamma_min = 1.e2;
double powerlaw_gamma_max = 1.e5;
double powerlaw_p = 3.25;

#ifdef DEBUG_WJET
struct wjet_debug_context {
  int valid;
  double X[NDIM];
  double rho;
  double uu;
  double Ne;
  double Thetae;
  double B_cgs;
  double sigma;
  double beta;
  int in_jet;
  int with_electrons;
  double sigma_transition;
  double constant_beta_e0;
  double constant_beta_e0_exponent;
  double jet_sigma_cut;
  double jet_beta_cut;
  double jet_thetae;
  double jet_ne_mult;
};

static struct wjet_debug_context wjet_ctx;
#pragma omp threadprivate(wjet_ctx)

void wjet_debug_update(const double X[NDIM], double rho, double uu, double Ne,
                       double Thetae, double B_cgs, double sigma, double beta,
                       int in_jet, int with_electrons, double sigma_transition,
                       double constant_beta_e0, double constant_beta_e0_exponent,
                       double jet_sigma_cut, double jet_beta_cut,
                       double jet_thetae, double jet_ne_mult)
{
  wjet_ctx.valid = 1;
  for (int mu = 0; mu < NDIM; mu++)
  {
    wjet_ctx.X[mu] = X[mu];
  }
  wjet_ctx.rho = rho;
  wjet_ctx.uu = uu;
  wjet_ctx.Ne = Ne;
  wjet_ctx.Thetae = Thetae;
  wjet_ctx.B_cgs = B_cgs;
  wjet_ctx.sigma = sigma;
  wjet_ctx.beta = beta;
  wjet_ctx.in_jet = in_jet;
  wjet_ctx.with_electrons = with_electrons;
  wjet_ctx.sigma_transition = sigma_transition;
  wjet_ctx.constant_beta_e0 = constant_beta_e0;
  wjet_ctx.constant_beta_e0_exponent = constant_beta_e0_exponent;
  wjet_ctx.jet_sigma_cut = jet_sigma_cut;
  wjet_ctx.jet_beta_cut = jet_beta_cut;
  wjet_ctx.jet_thetae = jet_thetae;
  wjet_ctx.jet_ne_mult = jet_ne_mult;
}

static void wjet_debug_zone_indices(const double X[NDIM],
                                    int *i_raw, int *j_raw, int *k_raw,
                                    int *i_clamped, int *j_clamped, int *k_clamped,
                                    double del[NDIM])
{
  double XG[NDIM] = { X[0], X[1], X[2], X[3] };
  double phi = XG[3];

  if (METRIC_eKS && METRIC_MKS3)
  {
    const double Xks[4] = { X[0], exp(X[1]), M_PI * X[2], X[3] };
    const double H0 = mks3H0;
    const double MY1 = mks3MY1;
    const double MY2 = mks3MY2;
    const double MP0 = mks3MP0;
    const double KSx1 = Xks[1];
    const double KSx2 = Xks[2];
    XG[0] = Xks[0];
    XG[1] = log(Xks[1] - mks3R0);
    XG[2] = (-(H0 * pow(KSx1, MP0) * M_PI) - pow(2., 1. + MP0) * H0 * MY1 * M_PI +
             2. * H0 * pow(KSx1, MP0) * MY1 * M_PI + pow(2., 1. + MP0) * H0 * MY2 * M_PI +
             2. * pow(KSx1, MP0) * atan(((-2. * KSx2 + M_PI) * tan((H0 * M_PI) / 2.)) / M_PI)) /
            (2. * H0 * (-pow(KSx1, MP0) - pow(2., 1 + MP0) * MY1 + 2. * pow(KSx1, MP0) * MY1 +
                        pow(2., 1. + MP0) * MY2) * M_PI);
    XG[3] = Xks[3];
  }

  if (stopx[3] > 0.0)
  {
    phi = fmod(XG[3], stopx[3]);
    if (phi < 0.0) phi += stopx[3];
  }

  *i_raw = (int)((XG[1] - startx[1]) / dx[1] - 0.5 + 1000) - 1000;
  *j_raw = (int)((XG[2] - startx[2]) / dx[2] - 0.5 + 1000) - 1000;
  *k_raw = (int)((phi  - startx[3]) / dx[3] - 0.5 + 1000) - 1000;

  Xtoijk(X, i_clamped, j_clamped, k_clamped, del);
}
#endif


void try_set_radiation_parameter(const char *line)
{
  read_param(line, "powerlaw_gamma_cut", &powerlaw_gamma_cut, TYPE_DBL);
  read_param(line, "powerlaw_gamma_min", &powerlaw_gamma_min, TYPE_DBL);
  read_param(line, "powerlaw_gamma_max", &powerlaw_gamma_max, TYPE_DBL);
  read_param(line, "powerlaw_p", &powerlaw_p, TYPE_DBL);
}

// determine w by finding effective w for total
// energy to match thermal (MJ) at Thetae
double kappa_w(double Thetae, double kappa)
{
  return (kappa - 3.)/kappa * Thetae;
}

// planck function
double Bnu_inv(double nu, double Thetae)
{
	double x = HPL * nu / (ME * CL * CL * Thetae);

	if (x < 1.e-3) { // Taylor expand if small
		return ((2. * HPL / (CL * CL)) /
			(x / 24. * (24. + x * (12. + x * (4. + x)))));
  }

	return (2. * HPL / (CL * CL)) / (exp(x) - 1.);
}

// return j_\nu/\nu^2, the invariant emissivity
double jnu_inv(double nu, double Thetae, double Ne, double B, double theta)
{
	double j = jnu(nu, Ne, Thetae, B, theta);

	return j / (nu * nu);
}

// return invariant scattering opacity if Compton scattering enabled
double alpha_inv_scatt(double nu, double Thetae, double Ne)
{
  #if COMPTON

	return nu * kappa_es(nu, Thetae) * Ne * MP;

  #else

  return 0.;

  #endif
}

// return invariant absorption opacity 
double alpha_inv_abs(double nu, double Thetae, double Ne, double B,
		     double theta)
{

#if BRESMSSTRAHLUNG && (MODEL_EDF==EDF_KAPPA_FIXED)
  fprintf(stderr, "ERROR absorptivities not set up for bremss and kappa!\n");
  exit(-1);
#endif

#if MODEL_EDF==EDF_KAPPA_FIXED

  // Pandya+ 2016 absorptivity
 
  double kap = model_kappa;
  double w = kappa_w(Thetae, kap);

  double nuc = EE*B/(2.*M_PI*ME*CL);
  double nuk = nuc*pow(w*kap,2)*sin(theta);
  double Xk = nu/nuk;

  double Aslo, Ashi, As;

  Aslo  = pow(Xk,-2./3.)*pow(3.,1./6.)*10./41.;
  Aslo *= 2.*M_PI/(pow(w*kap,10./3.-kap));
  Aslo *= (kap - 2.)*(kap - 1.)*kap/(3.*kap - 1.);
  Aslo *= gsl_sf_gamma(5./3.);
  // Evaluate 2F1(a,b;c,z), using analytic continuation if |z| > 1
  double a = kap - 1./3.;
  double b = kap + 1.;
  double c = kap + 2./3.;
  double z = -kap*w;
  double hg2F1;
  if (fabs(z) == 1.) {
    hg2F1 = 0.;
  } else if (fabs(z) < 1.) {
    hg2F1 = gsl_sf_hyperg_2F1(a, b, c, z);
  } else {
    hg2F1  = pow(1.-z,-a)*gsl_sf_gamma(c)*gsl_sf_gamma(b-a)/(gsl_sf_gamma(b)*gsl_sf_gamma(c-a))*gsl_sf_hyperg_2F1(a,c-b,a-b+1,1./(1.-z));
    hg2F1 += pow(1.-z,-b)*gsl_sf_gamma(c)*gsl_sf_gamma(a-b)/(gsl_sf_gamma(a)*gsl_sf_gamma(c-b))*gsl_sf_hyperg_2F1(b,c-a,b-a+1,1./(1.-z));
  }
  Aslo *= hg2F1;

  Ashi  = pow(Xk,-(1. + kap)/2.)*pow(M_PI,3./2.)/3.;
  Ashi *= (kap - 2.)*(kap - 1.)*kap/pow(w*kap,3.);
  Ashi *= (2.*gsl_sf_gamma(2. + kap/2.)/(2. + kap) - 1.);
  Ashi *= (pow(3./kap,19./4.) + 3./5.);

  double xbr = pow(-7./4. + 8./5.*kap,-43./50.);

  As = pow(pow(Aslo,-xbr) + pow(Ashi,-xbr),-1./xbr);
  double alphas = Ne*EE*EE/(nu*ME*CL)*As;
  double cut = exp(-nu/NUCUT);
  
  return nu*alphas*cut;

#elif MODEL_EDF==EDF_POWER_LAW

  /*
  double p = powerlaw_p;
  double gmin = powerlaw_gamma_min;
  double gmax = powerlaw_gamma_max;

  double sth = sin(theta);
  double nuc = EE * B / (2.*M_PI*ME*CL);
  double factor = (Ne * EE*EE)/(nu * ME*CL);

  double X = nu/(nuc*sth);

  double As = pow(3.,(p+1)/2.)*(p-1)/(4*(pow(gmin,1-p)-pow(gmax,1-p)));
  As *= gsl_sf_gamma((3*p+2)/12.)*gsl_sf_gamma((3*p+22)/12.)*pow(1./3.*X,-(p+2)/2.);

  return As*factor;
   */

  double sth = sin(theta);
  double nu_c = EE * B / (2 * M_PI * ME * CL); 

  double prefactor = Ne * EE*EE / (nu * ME * CL);

  double t1 = pow(3., (powerlaw_p+1)/2.) * (powerlaw_p - 1.);
  double t2 = 4. * (pow(powerlaw_gamma_min, 1.-powerlaw_p) - 
                    pow(powerlaw_gamma_max, 1.-powerlaw_p));
  double t3 = tgamma((3*powerlaw_p+2)/12.) * tgamma((3.*powerlaw_p+22)/12.);
  double t4 = pow(nu/(nu_c * sth), -(powerlaw_p+2)/2.);

  return nu * prefactor * t1 / t2 * t3 * t4;

#elif MODEL_EDF==EDF_MAXWELL_JUTTNER

	double j = jnu_inv(nu, Thetae, Ne, B, theta);
	double bnu = Bnu_inv(nu, Thetae);

  if (j > 0) {
	  return j / (bnu + 1.e-100);
  }

  return 0;

#else

  fprintf(stderr, "must select valid MODEL_EDF\n");
  exit(3);

#endif 
}


// return electron scattering opacity in cgs
double kappa_es(double nu, double Thetae)
{

	// assume pure hydrogen gas to
	// convert cross section to opacity
	
	double Eg = HPL * nu / (ME * CL * CL);

  if (Eg > 1.e75) {
    fprintf(stderr, "out of bounds: %g %g %g\n", Eg, Thetae, nu);
  }

	return total_compton_cross_lkup(Eg, Thetae) / MP;
}

// get frequency in fluid frame, in Hz
double get_fluid_nu(const double X[NDIM], const double K[NDIM], const double Ucov[NDIM],
                    const struct of_photon *ph, int nstep)
{
	// in electron rest-mass units 
	double energy = -(K[0]*Ucov[0] + K[1]*Ucov[1] + K[2]*Ucov[2] + K[3]*Ucov[3]);

  // in Hz
	double nu = energy * ME * CL * CL / HPL;

#ifdef DEBUG_WJET
  int bad = 0;
  double gcov[NDIM][NDIM];
  double gcon[NDIM][NDIM];
  double Ucon[NDIM] = {0.};
  double udotu = 0.0;
  gcov_func(X, gcov);
  gcon_func(gcov, gcon);

  MUNULOOP
  {
    if (IS_BAD(gcov[mu][nu]) || IS_BAD(gcon[mu][nu]))
    {
      bad = 1;
    }
  }

  MULOOP
  {
    if (IS_BAD(K[mu]) || IS_BAD(Ucov[mu]))
    {
      bad = 1;
    }
    for (int nu = 0; nu < NDIM; nu++)
    {
      Ucon[mu] += gcon[mu][nu] * Ucov[nu];
    }
    udotu += Ucon[mu] * Ucov[mu];
  }

  if (IS_BAD(udotu) || fabs(udotu + 1.0) > 1e-2)
  {
    bad = 1;
  }
  if (IS_BAD(energy) || IS_BAD(nu) || !(nu > 0.0))
  {
    bad = 1;
  }

  if (bad)
  {
    fprintf(stderr, "DEBUG_WJET get_fluid_nu: invalid state\n");
    if (ph)
    {
      fprintf(stderr,
              "ph_ptr=%p nstep=%d nscatt=%d w=%g E=%g E0=%g E0s=%g\n",
              (void *)ph, nstep, ph->nscatt, ph->w, ph->E, ph->E0, ph->E0s);
    }
    else
    {
      fprintf(stderr, "ph_ptr=(null) nstep=%d\n", nstep);
    }
    int i_raw = 0, j_raw = 0, k_raw = 0, i_clamped = 0, j_clamped = 0, k_clamped = 0;
    double del[NDIM] = {0.0};
    wjet_debug_zone_indices(X, &i_raw, &j_raw, &k_raw, &i_clamped, &j_clamped, &k_clamped, del);
    const int boundary_hit =
        (i_raw < 0 || j_raw < 0 || k_raw < 0 ||
         i_raw > N1 - 2 || j_raw > N2 - 2 || k_raw > N3 - 1 ||
         i_clamped <= 0 || j_clamped <= 0 || i_clamped >= N1 - 2 || j_clamped >= N2 - 2);
    fprintf(stderr, "X: %g %g %g %g\n", X[0], X[1], X[2], X[3]);
    fprintf(stderr, "zone raw=(%d,%d,%d) clamp=(%d,%d,%d) del=(%g,%g,%g) boundary_hit=%d\n",
            i_raw, j_raw, k_raw, i_clamped, j_clamped, k_clamped,
            del[1], del[2], del[3], boundary_hit);
    fprintf(stderr, "K: %g %g %g %g\n", K[0], K[1], K[2], K[3]);
    fprintf(stderr, "Ucov: %g %g %g %g\n", Ucov[0], Ucov[1], Ucov[2], Ucov[3]);
    fprintf(stderr, "Ucon: %g %g %g %g\n", Ucon[0], Ucon[1], Ucon[2], Ucon[3]);
    fprintf(stderr, "udotu=%g energy=%g nu=%g\n", udotu, energy, nu);
    fprintf(stderr,
            "gcov: %g %g %g %g %g %g %g %g %g %g\n",
            gcov[0][0], gcov[0][1], gcov[0][2], gcov[0][3],
            gcov[1][1], gcov[1][2], gcov[1][3],
            gcov[2][2], gcov[2][3],
            gcov[3][3]);
    fprintf(stderr, "gcon00=%g\n", gcon[0][0]);
    if (wjet_ctx.valid)
    {
      fprintf(stderr,
              "wjet_ctx: rho=%g uu=%g Ne=%g Thetae=%g B_cgs=%g sigma=%g beta=%g in_jet=%d\n",
              wjet_ctx.rho, wjet_ctx.uu, wjet_ctx.Ne, wjet_ctx.Thetae,
              wjet_ctx.B_cgs, wjet_ctx.sigma, wjet_ctx.beta, wjet_ctx.in_jet);
      fprintf(stderr,
              "wjet_params: with_electrons=%d sigma_transition=%g constant_beta_e0=%g constant_beta_e0_exponent=%g "
              "jet_sigma_cut=%g jet_beta_cut=%g jet_thetae=%g jet_ne_mult=%g\n",
              wjet_ctx.with_electrons, wjet_ctx.sigma_transition,
              wjet_ctx.constant_beta_e0, wjet_ctx.constant_beta_e0_exponent,
              wjet_ctx.jet_sigma_cut, wjet_ctx.jet_beta_cut,
              wjet_ctx.jet_thetae, wjet_ctx.jet_ne_mult);
      fprintf(stderr, "wjet_ctx_X: %g %g %g %g\n",
              wjet_ctx.X[0], wjet_ctx.X[1], wjet_ctx.X[2], wjet_ctx.X[3]);
    }
    else
    {
      fprintf(stderr, "wjet_ctx: unset\n");
    }
    // For DEBUG_WJET runs, report and allow caller to drop/skip photon.
    return -1.0;
  }
#endif

	if (IS_BAD(energy)) {
		fprintf(stderr, "isnan get_fluid_nu, K: %g %g %g %g\n",
			K[0], K[1], K[2], K[3]);
		fprintf(stderr, "isnan get_fluid_nu, X: %g %g %g %g\n",
			X[0], X[1], X[2], X[3]);
		fprintf(stderr, "isnan get_fluid_nu, U: %g %g %g %g\n",
			Ucov[0], Ucov[1], Ucov[2], Ucov[3]);
	}

	return nu;
}

// return angle between magnetic field and wavevector
double get_bk_angle(double X[NDIM], double K[NDIM], double Ucov[NDIM],
		    double Bcov[NDIM], double B)
{
	double k, mu;

	if (B == 0.) {
		return M_PI / 2.;
  }

	k = fabs(K[0]*Ucov[0] + K[1]*Ucov[1] + K[2]*Ucov[2] + K[3]*Ucov[3]);

	// B is in cgs but Bcov is in code units
	mu = (K[0] * Bcov[0] + K[1] * Bcov[1] + K[2] * Bcov[2] + K[3] * Bcov[3]) / (k * B / B_unit);

	if (fabs(mu) > 1.) {
		mu /= fabs(mu);
  }

	return acos(mu);

	(void)X; // silence unused parameter warning 
}
