"""Show that k=1 dominates the eval set for tournament n=4, making output accuracy misleading."""

import numpy as np

data_dir = "/dcs/large/u5514611/data"

for ds in [
    "Tournament_n4_Baseline_1e4_m1e7_b210",
    "Tournament_n4_TL_1e4_m1e7_b210",
    "Tournament_n4_ATL_1e4_m1e7_b210",
]:
    d = np.load(f"{data_dir}/{ds}/eval.npz")
    all_k = []
    for key in sorted(d.keys()):
        if "masked" in key and "output" in key:
            for row in d[key]:
                s, e = np.where(row == 211)[0][0], np.where(row == 217)[0][0]
                k = 0
                for digit in row[s + 1 : e]:
                    k = k * 210 + int(digit)
                all_k.append(k)

    all_k = np.array(all_k)
    k1 = np.sum(all_k == 1)
    print(f"{ds}:")
    print(f"  k=1: {k1}/{len(all_k)} ({100*k1/len(all_k):.1f}%)")
    print(f"  k>1: {len(all_k)-k1}/{len(all_k)} ({100*(len(all_k)-k1)/len(all_k):.1f}%)")
    print(f"  => A model that always predicts k=1 gets {100*k1/len(all_k):.1f}% output accuracy")
    print()

print(f"Theoretical: P(gcd of 4 random ints = 1) = 90/pi^4 = {90/np.pi**4:.1%}")
