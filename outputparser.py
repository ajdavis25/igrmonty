import re
import numpy as np
import matplotlib.pyplot as plt

logfile = "scatter_log.txt"

re_scatter = re.compile(
    r"SCATTER Thetae=(?P<Thetae>[\d\.e\+\-]+) "
    r"Ne=(?P<Ne>[\d\.e\+\-]+) "
    r"B=(?P<B>[\d\.e\+\-]+) "
    r"sigma=(?P<sigma>[\d\.e\+\-]+) "
    r"nscatt=(?P<nscatt>\d+)"
)

re_energy = re.compile(
    r"POST-SCATTER energy K0_before=(?P<before>[\d\.e\+\-]+)"
    r"\s+K0_after=(?P<after>[\d\.e\+\-]+)"
)

re_escape = re.compile(
    r"ESCAPE E=(?P<E>[\d\.e\+\-]+) "
    r"tau_scatt=(?P<tau_s>[\d\.e\+\-]+) "
    r"tau_abs=(?P<tau_a>[\d\.e\+\-]+) "
    r"nscatt=(?P<nscatt>\d+)"
)

Thetae_list = []
sigma_list = []
gain_list = []
escape_nscatt = []
escape_tau = []

with open(logfile) as f:
    for line in f:
        m = re_scatter.search(line)
        if m:
            Thetae_list.append(float(m.group("Thetae")))
            sigma_list.append(float(m.group("sigma")))
            continue

        m = re_energy.search(line)
        if m:
            before = float(m.group("before"))
            after = float(m.group("after"))
            gain_list.append(after / before if before > 0 else np.nan)
            continue

        m = re_escape.search(line)
        if m:
            escape_nscatt.append(int(m.group("nscatt")))
            escape_tau.append(float(m.group("tau_s")))
            continue

print("mean energy gain per scatter:", np.nanmean(gain_list))
print("median gain:", np.nanmedian(gain_list))
print("mean Thetae:", np.mean(Thetae_list))
print("median Thetae:", np.median(Thetae_list))
print("max sigma:", np.max(sigma_list))

plt.hist(gain_list, bins=40)
plt.xlabel("energy gain factor (K0_after / K0_before)")
plt.ylabel("count")
plt.title("compton energy gain distribution")
plt.show()

plt.hist(escape_nscatt, bins=20)
plt.xlabel("nscatt upon escape")
plt.ylabel("count")
plt.title("scattering count distribution at escape")
plt.show()

plt.hist(escape_tau, bins=40)
plt.xlabel("tau_scatt")
plt.ylabel("count")
plt.title("escape optical depth distribution")
plt.show()
