#include "decs.h"
#include "hotcross.h"
#include "compton.h"
#include "model_radiation.h"
#include <sys/stat.h>

static void minimal_bremss_coeffs(double Thetae, double *Fei, double *Fee_same)
{
  if (Thetae < 1.)
  {
    *Fei = 4. * sqrt(2. * Thetae / M_PI / M_PI / M_PI) *
           (1. + 1.781 * pow(Thetae, 1.34));
    *Fee_same = 20. / 9. / sqrt(M_PI) * (44. - 3. * M_PI * M_PI) *
                pow(Thetae, 1.5);
    *Fee_same *=
        (1. + 1.1 * Thetae + Thetae * Thetae - 1.25 * pow(Thetae, 2.5));
  }
  else
  {
    const double eta = 0.5616;
    *Fei = 9. * Thetae / (2. * M_PI) * (log(1.123 * Thetae + 0.48) + 1.5);
    *Fee_same = 24. * Thetae * (log(2. * eta * Thetae) + 1.28);
  }
}

static void check_pair_brems_channel(double Thetae, double Ne, double theta)
{
  double Fei = 0.;
  double Fee_same = 0.;
  double ei_strength;
  double ee_strength;
  double jb0, jb05, jb1;
  double minimal_ratio_05;
  double minimal_ratio_1;

  minimal_bremss_coeffs(Thetae, &Fei, &Fee_same);
  const double e_charge = 4.80e-10; // in esu
  const double re = e_charge * e_charge / ME / CL / CL;
  ei_strength = SIGMA_THOMSON * Fei;
  ee_strength = re * re * Fee_same;
  minimal_ratio_05 = (2.0 * ei_strength + 2.5 * ee_strength) /
                     (ei_strength + ee_strength);
  minimal_ratio_1 = (3.0 * ei_strength + 5.0 * ee_strength) /
                    (ei_strength + ee_strength);

  positron_ratio = 0.0;
  jb0 = jnu(1.e10, Ne, Thetae, 0.0, theta);
  positron_ratio = 0.5;
  jb05 = jnu(1.e10, Ne, Thetae, 0.0, theta);
  positron_ratio = 1.0;
  jb1 = jnu(1.e10, Ne, Thetae, 0.0, theta);

  if (!(jb0 > 0.0 && jb05 > jb0 && jb1 > jb05))
  {
    fprintf(stderr,
            "pair scaling test failed: brems emissivity not monotonic for Thetae=%g (%g, %g, %g)\n",
            Thetae, jb0, jb05, jb1);
    exit(1);
  }

  if (!((jb05 / jb0) > minimal_ratio_05 * (1. + 1.e-8)))
  {
    fprintf(stderr,
            "pair scaling test failed: missing e-e+ brems channel at Thetae=%g for f=0.5 (ratio=%g minimal=%g)\n",
            Thetae, jb05 / jb0, minimal_ratio_05);
    exit(1);
  }

  if (!((jb1 / jb0) > minimal_ratio_1 * (1. + 1.e-8)))
  {
    fprintf(stderr,
            "pair scaling test failed: missing e-e+ brems channel at Thetae=%g for f=1 (ratio=%g minimal=%g)\n",
            Thetae, jb1 / jb0, minimal_ratio_1);
    exit(1);
  }
}

void test_pair_scalings(void)
{
  double old_ratio = positron_ratio;
  const double nu = 2.3e11;
  const double Thetae = 10.0;
  const double Ne = 1.e7;
  const double B = 50.0;
  const double theta = M_PI / 3.0;

  fprintf(stderr, "testing pair scaling hooks (positron_ratio)\n");
  init_emiss_tables();

#if COMPTON
  init_hotcross();
  positron_ratio = 0.0;
  double a0 = alpha_inv_scatt(nu, Thetae, Ne);
  positron_ratio = 1.0;
  double a1 = alpha_inv_scatt(nu, Thetae, Ne);
  if (!(a0 > 0.0 && a1 > 0.0)) {
    fprintf(stderr, "pair scaling test failed: non-positive alpha_scatt (%g, %g)\n", a0, a1);
    exit(1);
  }
  double ratio_a = a1 / a0;
  double err = fabs(ratio_a - 3.0) / 3.0;
  if (err > 1.e-10) {
    fprintf(stderr, "pair scaling test failed: alpha_scatt ratio=%g expected=3\n", ratio_a);
    exit(1);
  }
#endif

  positron_ratio = 0.0;
  double j0 = jnu_inv(nu, Thetae, Ne, B, theta);
  positron_ratio = 1.0;
  double j1 = jnu_inv(nu, Thetae, Ne, B, theta);
  if (!(j1 > j0)) {
    fprintf(stderr, "pair scaling test failed: synch emissivity not increasing (%g -> %g)\n", j0, j1);
    exit(1);
  }

  check_pair_brems_channel(0.2, Ne, theta);
  check_pair_brems_channel(Thetae, Ne, theta);

  positron_ratio = old_ratio;
}

// test dNdgammae function in hotcross.c this is the only 
// public "interface" for the sampling eDF, so it can act
// as a sort of regression test
void test_hotcross_dNdgammae(const char *ofname, double Thetae)
{
  fprintf(stderr, "testing src/hotcross.c for Thetae=%g\n", Thetae);

  init_hotcross();

  FILE *fp = fopen(ofname, "w");
  fprintf(fp, "# %g %g %g\n", Thetae, model_kappa, powerlaw_p);

  double norm = getnorm_dNdg(Thetae);
  for (double lge = 0.; lge < 5; lge += 0.001) {
    fprintf(fp, "%g %g\n", lge, dNdgammae(Thetae, pow(10., lge)) * norm);
  }

  fprintf(fp, "\n");
  fclose(fp);
}

// test sample_beta_distr in compton.c. as above, this is the
// only public "interface", so we use it as a regression test
void test_compton_sample_beta_dist(const char *ofname, double Thetae)
{
  fprintf(stderr, "testing src/compton.c for Thetae=%g\n", Thetae);

  init_monty_rand(64);
  int nsamp = 100000;

  FILE *fp;
  double ge, be;

  fp = fopen(ofname, "w");
  fprintf(fp, "# %g %g %g\n", Thetae, model_kappa, powerlaw_p);

  for (int i=0; i<nsamp; ++i) {
    sample_beta_distr(Thetae, &ge, &be);
    fprintf(fp, "%g ", ge);
  }

  fprintf(fp, "\n");
  fclose(fp);
}


double Thetae_from_kappa_w(double kappa, double w)
{
  return w * kappa / (kappa - 3.);
}

// ---- Finding D6 regression tests (docs/2026-07-23_jet_implementation_changes.md
// sec 14): both Compton samplers checked against ground-truth numerical integrals
// of their target distributions. The KN cross section is reimplemented here on
// purpose (independent copy) so an accidental edit to the shared helper in
// compton.c cannot silently pass its own test.

static double test_sigma_hat(double K)
{
  if (K < 1.e-3)
    return 1. - 2. * K;
  return (3. / (4. * K * K)) *
         (2. + K * K * (1. + K) / ((1. + 2. * K) * (1. + 2. * K)) +
          (K * K - 2. * K - 2.) / (2. * K) * log(1. + 2. * K));
}

// sample_klein_nishina(k0) vs <k0p> from trapezoid integration of the code's own
// klein_nishina() differential cross section. Integrated on a grid uniform in
// log(eps): at large k0 the density's 1/eps part spreads its mass evenly across
// ~log(1+2k0)/log(10) decades, and a uniform-in-eps grid puts ~20% of that mass
// inside its first cell (this exact mistake made the first cluster run of this
// test, job 778171, reject a correct sampler at k0=1e5 -- see
// docs/2026-07-23_jet_implementation_changes.md sec 14).
static void check_kn_energy_sampler(double k0)
{
  const int NG = 20000;
  double eps0 = 1. / (1. + 2. * k0);
  double lmin = log(eps0);
  double num = 0., den = 0.;
  for (int i = 0; i <= NG; i++) {
    double eps = exp(lmin + (0. - lmin) * i / (double)NG);
    // extra factor eps = |d eps / d log(eps)| (Jacobian of the substitution)
    double w = klein_nishina(k0, eps * k0) * eps * ((i == 0 || i == NG) ? 0.5 : 1.0);
    num += w * eps;
    den += w;
  }
  double truth = (num / den) * k0;

  const int NS = 200000;
  double s = 0., s2 = 0.;
  for (int i = 0; i < NS; i++) {
    double v = sample_klein_nishina(k0);
    s += v;
    s2 += v * v;
  }
  double mean = s / NS;
  double se = sqrt(fmax(s2 / NS - mean * mean, 0.) / NS);
  double tol = 6. * se + 1.e-3 * truth;

  fprintf(stderr, "deep-KN test: kn_energy k0=%g sampled=%.6g truth=%.6g se=%.3g -> %s\n",
          k0, mean, truth, se, (fabs(mean - truth) <= tol) ? "ok" : "FAIL");
  if (fabs(mean - truth) > tol) {
    fprintf(stderr, "deep-KN test FAILED: sample_klein_nishina(k0=%g) mean off by %g (tol %g)\n",
            k0, fabs(mean - truth), tol);
    exit(1);
  }
}

// sample_electron_distr_p() vs E[gamma], E[K] from 2-D trapezoid integration of
// the target MJ(gamma) * (1-beta*mu) * sigma_KN(K), truncated to K <= 1e6 exactly
// as both sampler branches truncate. Which branch runs is decided inside
// sample_electron_distr_p by Thetae, so choosing Thetae on either side of 100
// exercises each branch against the same ground truth machinery.
static void check_electron_sampler(double Thetae, double k0)
{
  const int NLG = 2400, NMU = 4001;
  const double K0MAX_TEST = 1.e6;
  double lg_lo = 0., lg_hi = log(30. * Thetae);
  double dlg = (lg_hi - lg_lo) / NLG;
  double sw = 0., swg = 0., swk = 0.;
  for (int i = 0; i <= NLG; i++) {
    double ge = exp(lg_lo + i * dlg);
    if (ge <= 1.0000001) continue;
    double be = sqrt(1. - 1. / (ge * ge));
    double fg = fdist(ge, Thetae) * ((i == 0 || i == NLG) ? 0.5 : 1.0); // dN/dlog(gamma)
    for (int j = 0; j <= NMU - 1; j++) {
      double mu = -1. + 2. * j / (double)(NMU - 1);
      double K = ge * (1. - be * mu) * k0;
      if (!(K > 0.) || K > K0MAX_TEST) continue;
      double w = fg * (1. - be * mu) * test_sigma_hat(K) *
                 ((j == 0 || j == NMU - 1) ? 0.5 : 1.0);
      sw += w;
      swg += w * ge;
      swk += w * K;
    }
  }
  double truth_g = swg / sw, truth_k = swk / sw;

  const int NS = 100000;
  double k[NDIM] = {k0, k0, 0., 0.}; // null photon along x-hat
  double p[NDIM];
  double sg = 0., sg2 = 0., sk = 0., sk2 = 0.;
  for (int i = 0; i < NS; i++) {
    sample_electron_distr_p(k, p, Thetae);
    double ge = p[0];
    double K = k0 * (p[0] - p[1]); // = gamma*(1-beta*mu)*k0 for k along x-hat
    sg += ge; sg2 += ge * ge;
    sk += K;  sk2 += K * K;
  }
  double mg = sg / NS, seg = sqrt(fmax(sg2 / NS - mg * mg, 0.) / NS);
  double mk = sk / NS, sek = sqrt(fmax(sk2 / NS - mk * mk, 0.) / NS);
  double tg = 6. * seg + 0.01 * truth_g;
  double tk = 6. * sek + 0.01 * truth_k;

  fprintf(stderr, "deep-KN test: e-sampler Thetae=%g k0=%g: <gamma>=%.6g truth=%.6g | <K>=%.6g truth=%.6g -> %s\n",
          Thetae, k0, mg, truth_g, mk, truth_k,
          (fabs(mg - truth_g) <= tg && fabs(mk - truth_k) <= tk) ? "ok" : "FAIL");
  if (fabs(mg - truth_g) > tg || fabs(mk - truth_k) > tk) {
    fprintf(stderr, "deep-KN test FAILED: electron sampler at Thetae=%g k0=%g "
            "(dgamma=%g tol=%g; dK=%g tol=%g)\n",
            Thetae, k0, fabs(mg - truth_g), tg, fabs(mk - truth_k), tk);
    exit(1);
  }
}

void test_deep_kn_compton_sampling(void)
{
  // KN energy sampler: legacy branch (k0<1) and Butcher-Messel branch (k0>=1),
  // including a K0_MAX-scale energy.
  check_kn_energy_sampler(0.5);
  check_kn_energy_sampler(2.0);
  check_kn_energy_sampler(1.e5);

  // Electron sampler: legacy branch (Thetae<100), deep-KN branch (Thetae>=100),
  // and the exact regime that killed SLURM job 778136 (Thetae at the hard cap
  // against a multiply-scattered photon) -- which must now complete quickly.
  check_electron_sampler(80., 0.5);
  check_electron_sampler(150., 0.5);
  check_electron_sampler(1000., 2146.);

  fprintf(stderr, "deep-KN Compton sampling tests passed.\n");
}

void run_all_tests(void) {

  // Tests run before init_model(), so the RNG was previously uninitialized here
  // (latent segfault for every monty_rand-using test below). Deterministic seed
  // also makes test output reproducible.
  init_monty_rand(42);
  mkdir("test", 0755); // fopen("test/...") below fails silently if dir is missing

  // note: in the future, it might make sense to allow
  // switching of the eDF at runtime

  // set eDF parameters
  model_kappa = 4.;
  powerlaw_p = 3.;
  powerlaw_gamma_min = 25.;
  powerlaw_gamma_max = 1.e7;
  powerlaw_gamma_cut = 1.e3;

  // test hotcross.c functionality
  test_hotcross_dNdgammae("test/dNdgammae_0.1.out", 0.1);
  test_hotcross_dNdgammae("test/dNdgammae_1.out", 1);
  test_hotcross_dNdgammae("test/dNdgammae_5.out", 5);
  test_hotcross_dNdgammae("test/dNdgammae_10.out", 10);

  // test compton.c functionality
  test_compton_sample_beta_dist("test/sample_beta_0.1.out", 0.1);
  test_compton_sample_beta_dist("test/sample_beta_1.out", 1);
  test_compton_sample_beta_dist("test/sample_beta_5.out", 5);
  test_compton_sample_beta_dist("test/sample_beta_10.out", 10);

  test_pair_scalings();

  test_deep_kn_compton_sampling();

  exit(42);
}

// functions below left in for legacy reasons
// primarily used to unit test the individual
// components of the above. not guaranteed to
// compile/work.

void test_compton_sampling_functions(double kappa)
{
  fprintf(stderr, "testing eDF in compton.c for kappa=%g\n", kappa);

  model_kappa = kappa;

  FILE *fp = fopen("test/compton_sampling_functions.out", "w");
  fprintf(fp, "%g\n", kappa);

  // test to make sure the distribution functions return reasonable values
  for (double Thetae = 0; Thetae<100; Thetae+=0.5) {
    double geofmin = -1;
    double vofmin = 1.e10;
    double dofmin = 0;
    for (double ge=1.; ge < 1001; ge+=0.01) {
      double dist = fdist(ge, Thetae);
      double ddist = fabs(dfdgam(ge, &Thetae));
      if ( ddist < vofmin ) {
        geofmin = ge;
        vofmin = ddist;
        dofmin = dist;
      }
    }
    fprintf(fp, "%g %g %g %g\n", Thetae, geofmin, vofmin, dofmin);
  }

  fprintf(fp, "\n");
  fclose(fp);
}

void test_compton_sampling(double Thetae, double kappa, const char *fname)
{
  fprintf(stderr, "testing Compton sampling for Thetae=%g and kappa=%g\n", Thetae, kappa);

  model_kappa = kappa;

  FILE *fp = fopen(fname, "w");
  fprintf(fp, "%g %g ", kappa, Thetae);
  
  // draw monte carlo samples for some value of Thetae
  // check that the proper distribution is recovered
  init_monty_rand(64);
  double gamma_e, beta_e;
  for (int i=0; i<10000; ++i) {
    sample_beta_distr(Thetae, &gamma_e, &beta_e); 
    fprintf(fp, "%g ", gamma_e);
  }

  fprintf(fp, "\n");
  fclose(fp);
}

void test_emiss_abs()
{
  init_emiss_tables();

  model_kappa = 4.;

  fprintf(stderr, "DIST_KAPPA %d\n", MODEL_EDF==EDF_KAPPA_FIXED?1:0);

  double B = 10;
  double Thetae = 10;
  double theta = M_PI/3.;

  for (double lnu=9; lnu < 15; lnu += 0.2) {
    double nu = pow(10., lnu);
    double iem = jnu_inv(nu, Thetae, 1., B, theta);
    double iabs = alpha_inv_abs(nu, Thetae, 1., B, theta);
    if (1==0) 
    fprintf(stderr, "%g %g %g\n", nu, iem, iabs);
    double ijnu = int_jnu(1., Thetae, B, nu);
    fprintf(stderr, "%g %g\n", nu, ijnu);
  }
}


void test_hotcross() 
{
  init_hotcross();

  double w = 1.2;
  double thetae = 3.4;
  double value = total_compton_cross_lkup(w, thetae);

  fprintf(stderr, "%g %g -> %g\n", w, thetae, value);
}
