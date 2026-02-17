#include "decs.h"
#include "coordinates.h"

#define MAXNSTEP 1280000

static inline double sanitize_bias(double bias)
{
  if (!isfinite(bias) || bias < 1.)
    return 1.;
  return bias;
}

static inline double clamp_interval(double x, double lo, double hi)
{
  if (hi < lo)
  {
    const double mid = 0.5 * (lo + hi);
    lo = mid;
    hi = mid;
  }
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

static void debug_zone_indices(const double X[NDIM], int *i_raw, int *j_raw, int *k_raw,
                               int *i_clamped, int *j_clamped, int *k_clamped, double del[NDIM])
{
  double XG[NDIM] = {X[0], X[1], X[2], X[3]};
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

static int near_boundary_zone(int i_raw, int j_raw, int k_raw,
                              int i_clamped, int j_clamped, int k_clamped)
{
  if (i_raw < 0 || j_raw < 0 || k_raw < 0) return 1;
  if (i_raw > N1 - 2 || j_raw > N2 - 2 || k_raw > N3 - 1) return 1;
  if (i_clamped <= 0 || j_clamped <= 0) return 1;
  if (i_clamped >= N1 - 2 || j_clamped >= N2 - 2) return 1;
  if (k_clamped <= 0 || k_clamped >= N3 - 1) return 1;
  return 0;
}

static int try_boundary_recover_nu(struct of_photon *ph, int nstep,
                                   double gcov[NDIM][NDIM], double *Ne, double *Thetae, double *B,
                                   double Ucon[NDIM], double Ucov[NDIM],
                                   double Bcon[NDIM], double Bcov[NDIM],
                                   double *theta, double *nu)
{
  int i_raw, j_raw, k_raw, i_clamped, j_clamped, k_clamped;
  double del[NDIM] = {0.0};
  debug_zone_indices(ph->X, &i_raw, &j_raw, &k_raw, &i_clamped, &j_clamped, &k_clamped, del);

  if (!near_boundary_zone(i_raw, j_raw, k_raw, i_clamped, j_clamped, k_clamped))
  {
    return 0;
  }

  const double eps1 = 1.e-6 * fmax(fabs(dx[1]), 1.e-12);
  const double eps2 = 1.e-6 * fmax(fabs(dx[2]), 1.e-12);
  const double eps3 = 1.e-6 * fmax(fabs(dx[3]), 1.e-12);

  const double x1_lo = startx[1] + 0.5 * dx[1] + eps1;
  const double x1_hi = stopx[1]  - 0.5 * dx[1] - eps1;
  const double x2_lo = startx[2] + 0.5 * dx[2] + eps2;
  const double x2_hi = stopx[2]  - 0.5 * dx[2] - eps2;

  double Xfix[NDIM] = { ph->X[0], ph->X[1], ph->X[2], ph->X[3] };
  double Xold[NDIM] = { ph->X[0], ph->X[1], ph->X[2], ph->X[3] };

  if (i_raw <= 0 || i_clamped <= 0) Xfix[1] = x1_lo;
  if (i_raw >= N1 - 2 || i_clamped >= N1 - 2) Xfix[1] = x1_hi;
  if (j_raw <= 0 || j_clamped <= 0) Xfix[2] = x2_lo;
  if (j_raw >= N2 - 2 || j_clamped >= N2 - 2) Xfix[2] = x2_hi;
  Xfix[1] = clamp_interval(Xfix[1], x1_lo, x1_hi);
  Xfix[2] = clamp_interval(Xfix[2], x2_lo, x2_hi);

  if (stopx[3] > startx[3])
  {
    const double span = stopx[3] - startx[3];
    double phi = fmod(Xfix[3] - startx[3], span);
    if (phi < 0.0) phi += span;
    Xfix[3] = startx[3] + phi;

    if (k_raw < 0)
    {
      Xfix[3] = startx[3] + 0.5 * dx[3] + eps3;
    }
    else if (k_raw > N3 - 1)
    {
      Xfix[3] = stopx[3] - 0.5 * dx[3] - eps3;
    }
  }

  double gcov_fix[NDIM][NDIM], Ucon_fix[NDIM], Ucov_fix[NDIM], Bcon_fix[NDIM], Bcov_fix[NDIM];
  double Ne_fix = 0.0, Thetae_fix = 0.0, B_fix = 0.0, theta_fix = 0.0;

  gcov_func(Xfix, gcov_fix);
  get_fluid_params(Xfix, gcov_fix, &Ne_fix, &Thetae_fix, &B_fix, Ucon_fix, Ucov_fix, Bcon_fix, Bcov_fix);
  theta_fix = get_bk_angle(Xfix, ph->K, Ucov_fix, Bcov_fix, B_fix);
  const double nu_fix = get_fluid_nu(Xfix, ph->K, Ucov_fix, ph, nstep);

  if (!IS_BAD(nu_fix) && nu_fix > 0.0)
  {
    MULOOP
    {
      ph->X[mu] = Xfix[mu];
      Ucon[mu] = Ucon_fix[mu];
      Ucov[mu] = Ucov_fix[mu];
      Bcon[mu] = Bcon_fix[mu];
      Bcov[mu] = Bcov_fix[mu];
    }
    MUNULOOP gcov[mu][nu] = gcov_fix[mu][nu];
    *Ne = Ne_fix;
    *Thetae = Thetae_fix;
    *B = B_fix;
    *theta = theta_fix;
    *nu = nu_fix;
    init_dKdlam(ph->X, ph->K, ph->dKdlam);
#ifdef DEBUG_WJET
    fprintf(stderr,
            "DEBUG_WJET boundary_repair: recovered nu=%g at nstep=%d raw=(%d,%d,%d) clamp=(%d,%d,%d) del=(%g,%g,%g)\n",
            nu_fix, nstep, i_raw, j_raw, k_raw, i_clamped, j_clamped, k_clamped,
            del[1], del[2], del[3]);
    fprintf(stderr, "boundary_repair Xold=%g %g %g %g Xnew=%g %g %g %g\n",
            Xold[0], Xold[1], Xold[2], Xold[3], Xfix[0], Xfix[1], Xfix[2], Xfix[3]);
#endif
    return 1;
  }

  return 0;
}

#ifdef DEBUG_WJET
static long long debug_invalid_nu_dump_counter = 0;

static void debug_dump_invalid_nu_context(const char *stage, const struct of_photon *ph, int nstep,
                                          const double gcov[NDIM][NDIM], const double Ucon[NDIM],
                                          const double Ucov[NDIM], const double Bcon[NDIM],
                                          const double Bcov[NDIM], double Ne, double Thetae,
                                          double B, double nu, const double Xi[NDIM],
                                          const double Ki[NDIM], const double dKi[NDIM],
                                          double dl, double E0)
{
  long long dump_id = 0;
#pragma omp atomic capture
  dump_id = ++debug_invalid_nu_dump_counter;
  if (dump_id > 200)
  {
    if (dump_id == 201)
    {
      fprintf(stderr, "DEBUG_WJET invalid_nu: further detailed dumps suppressed after 200 events\n");
    }
    return;
  }

  int i_raw, j_raw, k_raw, i_clamped, j_clamped, k_clamped;
  double del[NDIM] = {0.0};
  debug_zone_indices(ph->X, &i_raw, &j_raw, &k_raw, &i_clamped, &j_clamped, &k_clamped, del);

  double r = 0.0, th = 0.0;
  bl_coord(ph->X, &r, &th);

  double gcov_local[NDIM][NDIM];
  double gcon[NDIM][NDIM];
  MUNULOOP gcov_local[mu][nu] = gcov[mu][nu];
  gcon_func(gcov_local, gcon);

  fprintf(stderr, "DEBUG_WJET invalid_nu_context[%lld]: stage=%s nstep=%d\n", dump_id, stage, nstep);
  fprintf(stderr, "X=%g %g %g %g r=%g th=%g nu=%g dl=%g E0=%g\n",
          ph->X[0], ph->X[1], ph->X[2], ph->X[3], r, th, nu, dl, E0);
  fprintf(stderr, "zone raw=(%d,%d,%d) clamp=(%d,%d,%d) del=(%g,%g,%g)\n",
          i_raw, j_raw, k_raw, i_clamped, j_clamped, k_clamped, del[1], del[2], del[3]);
  fprintf(stderr, "K=%g %g %g %g\n", ph->K[0], ph->K[1], ph->K[2], ph->K[3]);
  fprintf(stderr, "Ucon=%g %g %g %g\n", Ucon[0], Ucon[1], Ucon[2], Ucon[3]);
  fprintf(stderr, "Ucov=%g %g %g %g\n", Ucov[0], Ucov[1], Ucov[2], Ucov[3]);
  fprintf(stderr, "Bcon=%g %g %g %g\n", Bcon[0], Bcon[1], Bcon[2], Bcon[3]);
  fprintf(stderr, "Bcov=%g %g %g %g\n", Bcov[0], Bcov[1], Bcov[2], Bcov[3]);
  fprintf(stderr, "Ne=%g Thetae=%g B=%g\n", Ne, Thetae, B);
  fprintf(stderr, "gcon00=%g gcov00=%g\n", gcon[0][0], gcov_local[0][0]);
  fprintf(stderr,
          "gcov=%g %g %g %g %g %g %g %g %g %g\n",
          gcov_local[0][0], gcov_local[0][1], gcov_local[0][2], gcov_local[0][3],
          gcov_local[1][1], gcov_local[1][2], gcov_local[1][3],
          gcov_local[2][2], gcov_local[2][3], gcov_local[3][3]);

  if (Xi && Ki && dKi)
  {
    fprintf(stderr, "Xi=%g %g %g %g\n", Xi[0], Xi[1], Xi[2], Xi[3]);
    fprintf(stderr, "Ki=%g %g %g %g\n", Ki[0], Ki[1], Ki[2], Ki[3]);
    fprintf(stderr, "dKi=%g %g %g %g\n", dKi[0], dKi[1], dKi[2], dKi[3]);
  }

  if (isfinite(gcon[0][0]) && gcon[0][0] < 0.0)
  {
    double Uwork[NDIM], Bhat[NDIM] = {0.0, 1.0, 0.0, 0.0};
    double Econ[NDIM][NDIM], Ecov[NDIM][NDIM];
    double K_tetrad[NDIM], U_tetrad[NDIM], K_boosted[NDIM];
    MULOOP Uwork[mu] = Ucon[mu];

    if (isfinite(B) && fabs(B) > SMALL)
    {
      const double bnorm = B / B_unit;
      if (isfinite(bnorm) && fabs(bnorm) > SMALL)
      {
        MULOOP Bhat[mu] = Bcon[mu] / bnorm;
      }
    }

    make_tetrad(Uwork, Bhat, gcov_local, Econ, Ecov);
    coordinate_to_tetrad(Ecov, ph->K, K_tetrad);
    coordinate_to_tetrad(Ecov, Uwork, U_tetrad);
    boost(K_tetrad, U_tetrad, K_boosted);

    fprintf(stderr, "tetrad_input_U=%g %g %g %g\n", Uwork[0], Uwork[1], Uwork[2], Uwork[3]);
    fprintf(stderr, "tetrad_input_Bhat=%g %g %g %g\n", Bhat[0], Bhat[1], Bhat[2], Bhat[3]);
    fprintf(stderr,
            "Ecov00..03=%g %g %g %g Econ00..03=%g %g %g %g\n",
            Ecov[0][0], Ecov[0][1], Ecov[0][2], Ecov[0][3],
            Econ[0][0], Econ[0][1], Econ[0][2], Econ[0][3]);
    fprintf(stderr, "K_tetrad=%g %g %g %g\n", K_tetrad[0], K_tetrad[1], K_tetrad[2], K_tetrad[3]);
    fprintf(stderr, "U_tetrad=%g %g %g %g\n", U_tetrad[0], U_tetrad[1], U_tetrad[2], U_tetrad[3]);
    fprintf(stderr, "K_boosted=%g %g %g %g\n", K_boosted[0], K_boosted[1], K_boosted[2], K_boosted[3]);
  }
  else
  {
    fprintf(stderr, "tetrad/boost skipped: metric not usable (gcon00=%g)\n", gcon[0][0]);
  }
}
#endif

void track_super_photon(struct of_photon *ph)
{
  int bound_flag;
  double dtau_scatt, dtau_abs, dtau;
  double bi, bf;
  double alpha_scatti, alpha_scattf;
  double alpha_absi, alpha_absf;
  double dl, x1;
  double nu, Thetae, Ne, B, theta;
  struct of_photon php;
  double dtauK, frac;
  double bias = 1.;
  double Xi[NDIM], Ki[NDIM], dKi[NDIM], E0;
  double Gcov[NDIM][NDIM], Ucon[NDIM], Ucov[NDIM], Bcon[NDIM], Bcov[NDIM];
  int nstep = 0;

  // Don't track zero-weight photons
  if (ph->w < 1)
  {
    return;
  }

  // Quality control
  if (isnan(ph->X[0]) || isnan(ph->X[1]) || isnan(ph->X[2]) || isnan(ph->X[3]) ||
      isnan(ph->K[0]) || isnan(ph->K[1]) || isnan(ph->K[2]) || isnan(ph->K[3]))
  {
    fprintf(stderr, "track_super_photon: bad input photon.\n");
    fprintf(stderr,
            "X0,X1,X2,X3,K0,K1,K2,K3,w,nscatt: %g %g %g %g %g %g %g %g %g %d\n",
            ph->X[0], ph->X[1], ph->X[2], ph->X[3], ph->K[0],
            ph->K[1], ph->K[2], ph->K[3], ph->w, ph->nscatt);
    return;
  }

  dtauK = 2. * M_PI * L_unit / (ME * CL * CL / HBAR);

  // Initialize opacities
  gcov_func(ph->X, Gcov);
  get_fluid_params(ph->X, Gcov, &Ne, &Thetae, &B, Ucon, Ucov, Bcon, Bcov);

  theta = get_bk_angle(ph->X, ph->K, Ucov, Bcov, B);
  nu = get_fluid_nu(ph->X, ph->K, Ucov, ph, nstep);
  if (IS_BAD(nu) || !(nu > 0.0))
  {
    if (try_boundary_recover_nu(ph, nstep, Gcov, &Ne, &Thetae, &B, Ucon, Ucov, Bcon, Bcov, &theta, &nu))
    {
      // Proceed with recovered state.
    }
    else
    {
#ifdef DEBUG_WJET
      fprintf(stderr, "DEBUG_WJET track_super_photon: invalid nu after init\n");
      fprintf(stderr, "nu=%g X: %g %g %g %g\n", nu, ph->X[0], ph->X[1], ph->X[2], ph->X[3]);
      fprintf(stderr, "K: %g %g %g %g\n", ph->K[0], ph->K[1], ph->K[2], ph->K[3]);
      debug_dump_invalid_nu_context("init", ph, nstep, Gcov, Ucon, Ucov, Bcon, Bcov,
                                    Ne, Thetae, B, nu, NULL, NULL, NULL, 0.0, ph->E0s);
#endif
      ph->w = 0.0;
      return;
    }
  }
  alpha_scatti = alpha_inv_scatt(nu, Thetae, Ne);
  alpha_absi = alpha_inv_abs(nu, Thetae, Ne, B, theta);
  bi = sanitize_bias(bias_func(Thetae, ph->w));

  init_dKdlam(ph->X, ph->K, ph->dKdlam);
  while (!stop_criterion(ph))
  {
    // Save initial position/wave vector
    Xi[0] = ph->X[0];
    Xi[1] = ph->X[1];
    Xi[2] = ph->X[2];
    Xi[3] = ph->X[3];
    Ki[0] = ph->K[0];
    Ki[1] = ph->K[1];
    Ki[2] = ph->K[2];
    Ki[3] = ph->K[3];
    dKi[0] = ph->dKdlam[0];
    dKi[1] = ph->dKdlam[1];
    dKi[2] = ph->dKdlam[2];
    dKi[3] = ph->dKdlam[3];
    E0 = ph->E0s;

    // Evaluate stepsize
    dl = stepsize(ph->X, ph->K);

    // Step the geodesic
    push_photon(ph->X, ph->K, ph->dKdlam, dl, &(ph->E0s), 0);
    if (stop_criterion(ph))
      break;

#ifndef DEBUG_WJET
    if (IS_BAD(ph->X[0]) || IS_BAD(ph->X[1]) || IS_BAD(ph->X[2]) || IS_BAD(ph->X[3]) ||
        IS_BAD(ph->K[0]) || IS_BAD(ph->K[1]) || IS_BAD(ph->K[2]) || IS_BAD(ph->K[3]))
    {
      // Fail-safe: drop the photon if geodesic stepping produces NaNs.
      ph->w = 0.0;
      return;
    }
#endif

#ifdef DEBUG_WJET
    if (IS_BAD(ph->X[0]) || IS_BAD(ph->X[1]) || IS_BAD(ph->X[2]) || IS_BAD(ph->X[3]) ||
        IS_BAD(ph->K[0]) || IS_BAD(ph->K[1]) || IS_BAD(ph->K[2]) || IS_BAD(ph->K[3]))
    {
      fprintf(stderr, "DEBUG_WJET track_super_photon: NaN after push_photon\n");
      fprintf(stderr, "nstep=%d dl=%g E0=%g\n", nstep, dl, E0);
      fprintf(stderr, "X: %g %g %g %g\n", ph->X[0], ph->X[1], ph->X[2], ph->X[3]);
      fprintf(stderr, "K: %g %g %g %g\n", ph->K[0], ph->K[1], ph->K[2], ph->K[3]);
      fprintf(stderr, "Xi: %g %g %g %g\n", Xi[0], Xi[1], Xi[2], Xi[3]);
      fprintf(stderr, "Ki: %g %g %g %g\n", Ki[0], Ki[1], Ki[2], Ki[3]);
      fprintf(stderr, "dKi: %g %g %g %g\n", dKi[0], dKi[1], dKi[2], dKi[3]);
      exit(EXIT_FAILURE);
    }
#endif

#ifdef MODEL_TRANSPARENT
    if (0 == 1)
    {
#endif

      // Allow photon to interact with matter
      gcov_func(ph->X, Gcov);
      get_fluid_params(ph->X, Gcov, &Ne, &Thetae, &B, Ucon, Ucov, Bcon, Bcov);
      if (alpha_absi > 0. || alpha_scatti > 0. || Ne > 0.)
      {
        bound_flag = 0;
        if (Ne == 0.)
          bound_flag = 1;
        if (!bound_flag)
        {
          theta = get_bk_angle(ph->X, ph->K, Ucov, Bcov, B);
          nu = get_fluid_nu(ph->X, ph->K, Ucov, ph, nstep);
          if (IS_BAD(nu) || !(nu > 0.0))
          {
            if (try_boundary_recover_nu(ph, nstep, Gcov, &Ne, &Thetae, &B, Ucon, Ucov, Bcon, Bcov, &theta, &nu))
            {
              // Recovered from a boundary-adjacent invalid state.
            }
            else
            {
              fprintf(stderr, "isnan nu: track_super_photon dl,E0 %g %g\n", dl, E0);
              fprintf(stderr, "Xi, %g %g %g %g\n", Xi[0], Xi[1], Xi[2], Xi[3]);
              fprintf(stderr, "Ki, %g %g %g %g\n", Ki[0], Ki[1], Ki[2], Ki[3]);
              fprintf(stderr, "dKi, %g %g %g %g\n", dKi[0], dKi[1], dKi[2], dKi[3]);
#ifdef DEBUG_WJET
              fprintf(stderr, "DEBUG_WJET track_super_photon: dropping photon due to invalid nu\n");
              debug_dump_invalid_nu_context("step", ph, nstep, Gcov, Ucon, Ucov, Bcon, Bcov,
                                            Ne, Thetae, B, nu, Xi, Ki, dKi, dl, E0);
#else
              ph->w = 0.0;
              return;
#endif
              ph->w = 0.0;
              return;
            }
          }
        }

        // Scattering optical depth along step
        if (bound_flag || nu < 0.)
        {
          dtau_scatt = 0.5 * alpha_scatti * dtauK * dl;
          dtau_abs = 0.5 * alpha_absi * dtauK * dl;
          alpha_scatti = alpha_absi = 0.;
          bias = 1.;
          bi = 1.;
        }
        else
        {
          alpha_scattf = alpha_inv_scatt(nu, Thetae, Ne);
          dtau_scatt = 0.5 * (alpha_scatti + alpha_scattf) * dtauK * dl;
          alpha_scatti = alpha_scattf;

          // Absorption optical depth along step
          alpha_absf = alpha_inv_abs(nu, Thetae, Ne, B, theta);
          dtau_abs = 0.5 * (alpha_absi + alpha_absf) * dtauK * dl;
          alpha_absi = alpha_absf;

          bf = sanitize_bias(bias_func(Thetae, ph->w));
          bias = sanitize_bias(0.5 * (bi + bf));
          bi = bf;
        }

        x1 = -log(monty_rand());
        bias = sanitize_bias(bias);
        php.w = ph->w / bias;
        if (ph->ratio_brems < 0.9 && bias * dtau_scatt > x1 && php.w > WEIGHT_MIN)
        {
          php.tau_abs = ph->tau_abs + dtau_abs;
          php.tau_scatt = ph->tau_scatt + dtau_scatt;
          if (isnan(php.w) || isinf(php.w))
          {
            fprintf(stderr, "w isnan in track_super_photon: Ne, bias, ph->w, php.w  %g, %g, %g, %g\n",
                    Ne, bias, ph->w, php.w);
          }

          frac = x1 / (bias * dtau_scatt);

          // Apply absorption until scattering event
          dtau_abs *= frac;
          if (dtau_abs > 100)
            return; // This photon has been absorbed before scattering

          dtau_scatt *= frac;
          dtau = dtau_abs + dtau_scatt;
          if (dtau_abs < 1.e-3)
          {
            ph->w *= (1. - dtau / 24. * (24. - dtau * (12. - dtau * (4. - dtau))));
          }
          else
          {
            ph->w *= exp(-dtau);
          }

          // Interpolate position and wave vector to scattering event
          push_photon(Xi, Ki, dKi, dl * frac, &E0, 0);
          ph->X[0] = Xi[0];
          ph->X[1] = Xi[1];
          ph->X[2] = Xi[2];
          ph->X[3] = Xi[3];
          ph->K[0] = Ki[0];
          ph->K[1] = Ki[1];
          ph->K[2] = Ki[2];
          ph->K[3] = Ki[3];
          ph->dKdlam[0] = dKi[0];
          ph->dKdlam[1] = dKi[1];
          ph->dKdlam[2] = dKi[2];
          ph->dKdlam[3] = dKi[3];
          ph->E0s = E0;

          // Get plasma parameters at new position
          gcov_func(ph->X, Gcov);
          get_fluid_params(ph->X, Gcov, &Ne, &Thetae, &B, Ucon, Ucov, Bcon, Bcov);

          // Actually about to scatter photon
          if (Ne > 0.)
          {
            if (!isfinite(bias) || bias < 1.0)
            {
#pragma omp atomic
              ++invalid_bias;
              bias = 1.0;
              php.w = ph->w;
            }
            scatter_super_photon(ph, &php, Ne, Thetae, B, Ucon, Bcon, Gcov);

            if (ph->w < 1.e-100)
            { // Possible problem while enforcing k.k = 0
              return;
            }
            track_super_photon(&php);
          }

          theta = get_bk_angle(ph->X, ph->K, Ucov, Bcov, B);
          nu = get_fluid_nu(ph->X, ph->K, Ucov, ph, nstep);
          if (nu < 0.)
          {
            alpha_scatti = alpha_absi = 0.;
          }
          else
          {
            alpha_scatti = alpha_inv_scatt(nu, Thetae, Ne);
            alpha_absi = alpha_inv_abs(nu, Thetae, Ne, B, theta);
          }
          bi = sanitize_bias(bias_func(Thetae, ph->w));

          ph->tau_abs += dtau_abs;
          ph->tau_scatt += dtau_scatt;
        }
        else
        {
          if (dtau_abs > 100)
            return; // This photon has been absorbed
          ph->tau_abs += dtau_abs;
          ph->tau_scatt += dtau_scatt;
          dtau = dtau_abs + dtau_scatt;
          if (dtau < 1.e-3)
            ph->w *= (1. - dtau / 24. * (24. - dtau * (12. - dtau * (4. - dtau))));
          else
            ph->w *= exp(-dtau);
        }
      }

#ifdef MODEL_TRANSPARENT
    }
#endif

    nstep++;

    // Signs that something's wrong with the integration
    if (nstep > MAXNSTEP)
    {
      fprintf(stderr, "X1,X2,K1,K2,bias: %g %g %g %g %g\n", ph->X[1], ph->X[2],
              ph->K[1], ph->K[2], bias);
      break;
    }
  }

  // Accumulate result in spectrum on escape
  if (record_criterion(ph) && nstep < MAXNSTEP)
  {
#pragma omp atomic
    N_scatt += ph->nscatt;

#pragma omp critical(MAXTAU)
    {
      if (ph->tau_scatt > max_tau_scatt)
        max_tau_scatt = ph->tau_scatt;
    }

    if (record_photons)
      record_super_photon(ph);
  }
}
#undef MAXNSTEP
