#include "decs.h"
#include "hotcross.h"
#include "compton.h"
#include "model_radiation.h"

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

void run_all_tests(void) {

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
