"""
Этап 6 (Б): сравнение изотопов — мощность, срок службы, пригодность, доступность.

Для каждого изотопа из первых принципов:
  • удельная активность A = ln2·N_A/(T½·M) и удельная мощность (Вт/г чистого изотопа);
  • мощность на объём в рабочей химической форме (оксид, металл, гидрид);
  • энергия, отданная за 10 и 50 лет (с учётом распада);
  • радиационная стойкость: T_max передачи атому против порога смещения Ed;
  • мощность алмазной стопки по закону тонких слоёв (этап 5б):
        P = P_ист·η_conv·x/((1+x)(1+k·x)),  k = S_ист/S_алмаз;
  • напряжение прямого сбора полем (этап 4): U ≈ E_max/q.
Данные ядерной физики — справочные (NNDC/ENSDF), плотности форм — приближённые.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NA, Q, YEAR, MEC2, AMU = 6.022e23, 1.602e-19, 3.156e7, 0.511, 931.494
ETA_CONV = 0.245                                    # реалистичный алмазный диод (этапы 2–5)
ED_DIAMOND, ED_SIC = 37.0, 20.0                     # эВ, пороги смещения (этап 2)

# T½ (лет), M (г/моль), тип, Ē — средняя энергия частицы на распад (кэВ), E_max (кэВ),
# форма: плотность (г/см³), массовая доля изотопа, (Z, A) — средние на атом формы
ISO = {
    "H-3":    dict(T=12.32,  M=3.016,  kind="β", Em=5.69,   Emax=18.6,
                   form="TiT₂",   rho=3.75, w=0.111, ZA=(8.0, 18.0)),
    "C-14":   dict(T=5700,   M=14.003, kind="β", Em=49.5,   Emax=156.5,
                   form="алмаз ¹⁴C", rho=4.10, w=1.0, ZA=(6, 14.0)),
    "Ni-63":  dict(T=100.1,  M=62.93,  kind="β", Em=17.4,   Emax=66.9,
                   form="металл", rho=8.9, w=1.0, ZA=(28, 63.0)),
    "Kr-85":  dict(T=10.76,  M=84.91,  kind="β", Em=251,    Emax=687,
                   form="газ 100 бар", rho=0.34, w=1.0, ZA=(36, 85.0)),
    "Pm-147": dict(T=2.62,   M=146.9,  kind="β", Em=62.0,   Emax=224.6,
                   form="Pm₂O₃", rho=6.85, w=0.86, ZA=(29.2, 68.4)),
    "Sr-90":  dict(T=28.79,  M=89.91,  kind="β", Em=196+935, Emax=2280,   # + дочерний Y-90
                   form="SrTiO₃", rho=5.12, w=0.484, ZA=(16.8, 37.2)),
    "Po-210": dict(T=0.379,  M=209.98, kind="α", Em=5304,   Emax=5304,
                   form="металл", rho=9.2, w=1.0, ZA=(84, 210.0)),
    "Pu-238": dict(T=87.7,   M=238.05, kind="α", Em=5490,   Emax=5499,
                   form="PuO₂", rho=11.46, w=0.881, ZA=(36.7, 90.0)),
    "Am-241": dict(T=432.6,  M=241.06, kind="α", Em=5480,   Emax=5486,
                   form="AmO₂", rho=11.68, w=0.883, ZA=(37.0, 91.0)),
    "Cm-244": dict(T=18.11,  M=244.06, kind="α", Em=5795,   Emax=5805,
                   form="Cm₂O₃", rho=11.7, w=0.91, ZA=(43.2, 107.2)),
}
# Доступность и особенности (качественно, по открытым источникам)
NOTES = {
    "H-3":    "побочный продукт тяжеловодных реакторов (CANDU); газ, нужен гидрид металла",
    "C-14":   "есть в облучённом реакторном графите (тонны в Великобритании); алмаз из самого источника",
    "Ni-63":  "облучение Ni-62 в реакторе; обогащение обычно 15–20%, дорого",
    "Kr-85":  "осколок деления, выделяется при переработке ОЯТ; газ; γ 514 кэВ (0,4%)",
    "Pm-147": "осколок деления; батареи кардиостимуляторов «Betacel» в 1970-х; живёт мало",
    "Sr-90":  "много в ОЯТ; советские РИТЭГи «Бета-М»; сильное тормозное излучение",
    "Po-210": "облучение Bi-209; нагреватели «Лунохода»; живёт 138 дней",
    "Pu-238": "очень дефицитен (NASA — порядка кг в год); РИТЭГи",
    "Am-241": "выделяется из старого плутония; программа РИТЭГ ESA; γ 59,5 кэВ",
    "Cm-244": "есть в ОЯТ; спонтанное деление → нейтроны, тяжёлая защита",
}

def Tmax_eV(E_keV, M_amu):                          # электрон → атом (этап 2)
    E = E_keV/1000
    return 2*E*(E + 2*MEC2)/(M_amu*AMU)*1e6

def Tmax_alpha_eV(E_keV, M_amu, m=4.0026):          # альфа → атом, упругое столкновение
    return 4*m*M_amu/(m + M_amu)**2*E_keV*1e3

def S_rel(rho, Z, A, E_keV):                        # тормозная способность электронов (Бете), отн. ед.
    J = (9.76*Z + 58.5*Z**-0.19)*1e-3
    return rho*Z/A*np.log(1.166*(E_keV + 0.85*J)/J)

S_DIA = lambda E: S_rel(3.52, 6, 12.01, E)

rows = []
for n, d in ISO.items():
    SA = np.log(2)*NA/(d["T"]*YEAR*d["M"])          # Бк/г
    Pg = SA*d["Em"]*1e3*Q                            # Вт/г чистого изотопа
    Pv = Pg*d["rho"]*d["w"]                          # Вт/см³ рабочей формы
    tau = d["T"]/np.log(2)
    E10 = Pg*tau*(1 - 2**(-10/d["T"]))*8766          # Вт·ч/г
    E50 = Pg*tau*(1 - 2**(-50/d["T"]))*8766
    if d["kind"] == "β":
        Tm = Tmax_eV(d["Emax"], 12.01)
        k = S_rel(d["rho"], *d["ZA"], d["Em"])/S_DIA(d["Em"])
        x9 = (1/0.9 - 1)/k
        Pstack = Pv*ETA_CONV*0.9*x9/(1 + x9)          # компромисс (90% от η_conv)
        xo = 1/np.sqrt(k)
        Pmax = Pv*ETA_CONV*xo/((1 + xo)*(1 + k*xo))
        U = d["Emax"]
    else:
        Tm = Tmax_alpha_eV(d["Emax"], 12.01); k = np.nan; Pstack = Pmax = 0.0
        U = d["Emax"]/2
    rows.append(dict(name=n, SA=SA/3.7e10, Pg=Pg, Pv=Pv, E10=E10, E50=E50, Tm=Tm, k=k,
                     ok_dia=Tm < ED_DIAMOND, ok_sic=Tmax_eV(d["Emax"], 12.01) < ED_SIC if d["kind"] == "β" else False,
                     Pstack=Pstack, Pmax=Pmax, U=U, P20=2**(-20/d["T"])))

if __name__ == "__main__":
    print(f"{'изотоп':8}{'T½,лет':>9}{'Ки/г':>9}{'Вт/г':>10}{'Вт/см³':>9}{'Вт·ч/г 10л':>12}{'50л':>9}"
          f"{'P(20л)':>8}{'алмаз':>7}{'SiC':>5}")
    for r, (n, d) in zip(rows, ISO.items()):
        print(f"{n:8}{d['T']:9.3g}{r['SA']:9.3g}{r['Pg']:10.3g}{r['Pv']:9.3g}{r['E10']:12.3g}{r['E50']:9.3g}"
              f"{r['P20']*100:7.0f}%{'  ok' if r['ok_dia'] else '  !!':>7}{'  ok' if r['ok_sic'] else '  !!':>5}")

    print("\nАлмазная стопка по закону тонких слоёв (реалистичный диод η_conv = 24,5%),"
          " только β без повреждения алмаза помечены ✓")
    print(f"{'изотоп':8}{'форма':>12}{'k':>6}{'P компр., мВт/см³':>19}{'P макс.':>9}{'U поля, кВ':>12}")
    for r, (n, d) in zip(rows, ISO.items()):
        if d["kind"] == "β":
            print(f"{n:8}{d['form']:>12}{r['k']:6.2f}{r['Pstack']*1e3:19.3g}{r['Pmax']*1e3:9.3g}"
                  f"{r['U']:12.0f}  {'✓' if r['ok_dia'] else '✗ повреждает алмаз'}")
        else:
            print(f"{n:8}{d['form']:>12}{'—':>6}{'α разрушает диод':>19}{'':9}{r['U']:12.0f}  только поле")

    print("\nМасса чистого изотопа на 1 мВт электрической мощности через 20 лет"
          " (стопка: η ≈ 22%; поле: η ≈ 67%)")
    for r, (n, d) in zip(rows, ISO.items()):
        eta = 0.22 if (d["kind"] == "β" and r["ok_dia"]) else 0.67
        route = "стопка" if eta == 0.22 else "поле"
        if r["P20"] < 1e-3:
            print(f"  {n:8} {'—':>10}    (распадается раньше 20 лет)")
            continue
        print(f"  {n:8} {1e-3/(r['Pg']*r['P20']*eta):10.3g} г  ({route})")

    print("\nДоступность:")
    for n, t in NOTES.items():
        print(f"  {n:8} {t}")

    # --- графики
    fig, ax = plt.subplots(3, 1, figsize=(7.5, 14))
    for r, (n, d) in zip(rows, ISO.items()):
        c = "tab:blue" if d["kind"] == "β" else "tab:red"
        ax[0].scatter(d["T"], r["Pg"], c=c, s=50)
        ax[0].annotate(n, (d["T"], r["Pg"]), xytext=(5, 3), textcoords="offset points", fontsize=8)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("Период полураспада, лет"); ax[0].set_ylabel("Удельная мощность, Вт/г")
    ax[0].set_title("Мощность против срока службы (синие — β, красные — α)"); ax[0].grid(alpha=.3, which="both")

    names = [r["name"] for r in rows]; x = np.arange(len(names))
    ax[1].bar(x - 0.2, [r["E10"] for r in rows], 0.4, label="за 10 лет")
    ax[1].bar(x + 0.2, [r["E50"] for r in rows], 0.4, label="за 50 лет")
    ax[1].set_yscale("log"); ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=30, fontsize=8)
    ax[1].set_ylabel("Энергия распада, Вт·ч на грамм"); ax[1].axhline(0.25, ls="--", c="gray")
    ax[1].text(len(names) - 3.2, 0.3, "Li-ion ≈ 0,25 Вт·ч/г", fontsize=8)
    ax[1].set_title("Запас энергии на грамм чистого изотопа"); ax[1].grid(alpha=.3, axis="y"); ax[1].legend(fontsize=8)

    for r, (n, d) in zip(rows, ISO.items()):
        if d["kind"] != "β":
            continue
        yrs = np.linspace(0, 50, 200)
        ax[2].plot(yrs, r["Pstack"]*1e3*2**(-yrs/d["T"]), label=n + ("" if r["ok_dia"] else " (повреждает алмаз)"),
                   ls="-" if r["ok_dia"] else "--")
    ax[2].set_yscale("log"); ax[2].set_xlabel("Годы работы"); ax[2].set_ylabel("мВт/см³ (компромисс)")
    ax[2].set_title("Алмазная стопка: мощность во времени"); ax[2].grid(alpha=.3, which="both"); ax[2].legend(fontsize=7)
    plt.tight_layout(); plt.savefig("isotopes.png", dpi=130)
