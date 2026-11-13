"""Genera las figuras de las slides (carpeta img/).

Correr desde esta carpeta:  python generar_figuras.py
Usa los mismos datasets que las notebooks (carpeta data/).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Paleta de las slides de la materia (verde salvia / naranja / verde oscuro), siempre en este orden
PAL = ["#6e8f6a", "#b85c2e", "#1d3b2a", "#c49a2c", "#6f8fae", "#a8728f", "#8a8a80", "#d08c60"]
GRIS = "#6b7064"
TINTA = "#20241e"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.family": ["Lato", "DejaVu Sans"],
    "font.size": 15, "axes.titlesize": 15, "axes.labelsize": 14,
    "axes.titlecolor": TINTA, "text.color": TINTA,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#20241e", "axes.linewidth": 0.8, "axes.labelcolor": TINTA,
    "xtick.color": TINTA, "ytick.color": TINTA, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "axes.grid": False,
    "legend.frameon": False,
})


def guardar(fig, nombre):
    fig.savefig(f"img/{nombre}.png", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1) K-MEANS: clientes de un shopping
# ---------------------------------------------------------------------------
mall = pd.read_csv("data/mall_customers.csv").rename(columns={
    "Annual Income (k$)": "ingreso", "Spending Score (1-100)": "score_gasto", "Age": "edad"})
X = mall[["ingreso", "score_gasto"]].values
scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)

# Datos crudos
fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.scatter(mall.ingreso, mall.score_gasto, s=28, color=GRIS, alpha=0.75, edgecolor="white", linewidth=0.6)
ax.set_xlabel("Ingreso anual (miles de USD)")
ax.set_ylabel("Score de gasto (1-100)")
ax.set_title("200 clientes de un shopping: ¿ves grupos?")
guardar(fig, "km_datos")

# Paso a paso (implementación manual para mostrar las iteraciones)
rng = np.random.default_rng(3)
K = 5
cent = Xs[rng.choice(len(Xs), K, replace=False)]
estados = []
for it in range(10):
    d = ((Xs[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
    lab = d.argmin(1)
    estados.append((cent.copy(), lab.copy()))
    nuevo = np.array([Xs[lab == k].mean(0) for k in range(K)])
    if np.allclose(nuevo, cent):
        break
    cent = nuevo
mostrar = [(0, "1. Centroides al azar"), (0, "2. Cada cliente al más cercano"),
           (1, "3. Muevo centroides al promedio"), (len(estados) - 1, f"4. Repito hasta que no cambia ({len(estados)} iter.)")]
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharex=True, sharey=True)
for i, (ax, (idx, tit)) in enumerate(zip(axes, mostrar)):
    c, l = estados[idx]
    if i == 0:
        ax.scatter(Xs[:, 0], Xs[:, 1], s=14, color="#cfc8b6")
    else:
        ax.scatter(Xs[:, 0], Xs[:, 1], s=14, c=[PAL[k] for k in l], alpha=0.8)
    ax.scatter(c[:, 0], c[:, 1], marker="X", s=190, color="black", edgecolor="white", linewidth=1.5)
    ax.set_title(tit, fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
axes[0].set_ylabel("Score de gasto")
for ax in axes:
    ax.set_xlabel("Ingreso")
guardar(fig, "km_pasos")

# Elegir K: codo + silhouette (dos gráficos, no doble eje)
ks = range(1, 11)
inercias, sils = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    inercias.append(km.inertia_)
    sils.append(silhouette_score(Xs, km.labels_) if k > 1 else np.nan)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].plot(ks, inercias, marker="o", color=PAL[0], lw=2)
axes[0].annotate("codo", (5, inercias[4]), xytext=(6.5, inercias[4] + 90),
                 arrowprops=dict(arrowstyle="->", color=GRIS), color=GRIS)
axes[0].set_title("Método del codo: inercia")
axes[0].set_xlabel("K (cantidad de grupos)"); axes[0].set_ylabel("Inercia")
axes[1].plot(ks, sils, marker="o", color=PAL[2], lw=2)
axes[1].scatter([5], [sils[4]], s=160, facecolor="none", edgecolor="black", lw=1.5)
axes[1].set_title("Silhouette (más alto = mejor)")
axes[1].set_xlabel("K (cantidad de grupos)"); axes[1].set_ylabel("Silhouette promedio")
for ax in axes:
    ax.set_xticks(list(ks))
guardar(fig, "km_elegir_k")

# Segmentos finales con nombre
km = KMeans(n_clusters=5, n_init=10, random_state=42).fit(Xs)
mall["cluster"] = km.labels_
cent = scaler.inverse_transform(km.cluster_centers_)


def nombre_segmento(ing, gas):
    if ing > 70 and gas > 60: return "VIP"
    if ing > 70 and gas < 40: return "Ahorradores con plata"
    if ing < 40 and gas > 60: return "Gastadores impulsivos"
    if ing < 40 and gas < 40: return "Cuidadosos"
    return "Clientes promedio"


orden = ["Clientes promedio", "VIP", "Ahorradores con plata", "Gastadores impulsivos", "Cuidadosos"]
nombres = {k: nombre_segmento(*cent[k]) for k in range(5)}
fig, ax = plt.subplots(figsize=(7.4, 4.8))
for k in sorted(range(5), key=lambda k: orden.index(nombres[k])):
    m = mall.cluster == k
    ax.scatter(mall.ingreso[m], mall.score_gasto[m], s=30, color=PAL[orden.index(nombres[k])],
               edgecolor="white", linewidth=0.6, label=f"{nombres[k]} ({m.sum()})")
    ax.scatter(*cent[k], marker="X", s=200, color="black", edgecolor="white", linewidth=1.5)
ax.set_xlabel("Ingreso anual (miles de USD)")
ax.set_ylabel("Score de gasto (1-100)")
ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), title="Segmento (n)")
ax.set_title("K = 5 segmentos")
guardar(fig, "km_segmentos")

# ---------------------------------------------------------------------------
# 2) JERÁRQUICO: países
# ---------------------------------------------------------------------------
# Ejemplo de juguete: 7 puntos con nombre
toy = pd.DataFrame({"x": [1, 1.4, 4, 4.5, 4.2, 8, 8.6], "y": [1, 1.6, 4, 3.6, 4.6, 1.2, 1.8]},
                   index=list("ABCDEFG"))
Z = linkage(toy.values, "ward")
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(toy.x, toy.y, s=120, color=PAL[0], zorder=3)
for n, r in toy.iterrows():
    axes[0].annotate(n, (r.x, r.y), xytext=(7, 5), textcoords="offset points", fontsize=13, weight="bold")
axes[0].set_title("7 puntos"); axes[0].set_xlim(0, 9.8); axes[0].set_ylim(0, 5.5)
dendrogram(Z, labels=list(toy.index), ax=axes[1], color_threshold=0, above_threshold_color=GRIS)
axes[1].axhline(4, color=PAL[1], ls="--", lw=1.8)
axes[1].text(axes[1].get_xlim()[1], 4.2, "corte → 3 grupos", ha="right", color=PAL[1])
axes[1].set_title("Dendrograma: quién se unió con quién, y a qué distancia")
axes[1].set_ylabel("Distancia al unirse"); axes[1].grid(False)
guardar(fig, "jer_toy")

# Paso a paso: cómo se arma el dendrograma, unión por unión
from matplotlib.patches import Ellipse
dn = dendrogram(Z, labels=list(toy.index), no_plot=True)
orden_hojas = dn["ivl"]
xpos = {n: 5 + 10 * i for i, n in enumerate(orden_hojas)}
alturas = list(Z[:, 2])
segmentos = {}
for ic, dc in zip(dn["icoord"], dn["dcoord"]):
    fila = int(np.argmin([abs(h - dc[1]) for h in alturas]))
    segmentos[fila] = (ic, dc)
miembros = {i: [n] for i, n in enumerate(toy.index)}
for i, fila in enumerate(Z):
    miembros[len(toy) + i] = miembros[int(fila[0])] + miembros[int(fila[1])]


def dibujar_grupo(ax, nombres, color, lw):
    pts = toy.loc[nombres][["x", "y"]].values
    c = pts.mean(0)
    w = max(np.ptp(pts[:, 0]), 0) + 0.9
    h = max(np.ptp(pts[:, 1]), 0) + 0.9
    ax.add_patch(Ellipse(c, w * 1.25, h * 1.25, fill=False, color=color, lw=lw))


for paso in range(1, len(Z) + 1):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.scatter(toy.x, toy.y, s=120, color=PAL[0], zorder=3)
    for n, r in toy.iterrows():
        ax.annotate(n, (r.x, r.y), xytext=(7, 5), textcoords="offset points", fontsize=13, weight="bold")
    vivos = set(range(len(toy)))
    for i in range(paso):
        a, b = int(Z[i, 0]), int(Z[i, 1])
        vivos -= {a, b}
        vivos.add(len(toy) + i)
    for g in vivos:
        if len(miembros[g]) > 1 and g != len(toy) + paso - 1:
            dibujar_grupo(ax, miembros[g], GRIS, 1.2)
    nuevo = miembros[len(toy) + paso - 1]
    if len(nuevo) < len(toy):
        dibujar_grupo(ax, nuevo, PAL[1], 2.5)
    ax.set_xlim(-0.5, 10.5); ax.set_ylim(-0.8, 6.3)
    a, b = int(Z[paso - 1, 0]), int(Z[paso - 1, 1])
    txt_a, txt_b = "+".join(miembros[a]), "+".join(miembros[b])
    ax.set_title(f"Paso {paso}: junto {{{txt_a}}} con {{{txt_b}}}", color=PAL[1] if True else None)
    ax = axes[1]
    for i in range(paso):
        ic, dc = segmentos[i]
        ax.plot(ic, dc, color=PAL[1] if i == paso - 1 else GRIS, lw=2.5 if i == paso - 1 else 1.5)
    ax.set_xticks([xpos[n] for n in orden_hojas], orden_hojas, fontsize=12)
    ax.set_xlim(0, 10 * len(toy)); ax.set_ylim(0, Z[-1, 2] * 1.1)
    ax.set_ylabel("Distancia al unirse"); ax.grid(False)
    ax.set_title("Dendrograma (se arma de abajo hacia arriba)")
    ax.annotate(f"altura = {Z[paso-1, 2]:.1f}", (np.mean(segmentos[paso-1][0][1:3]), Z[paso-1, 2]),
                xytext=(0, 8), textcoords="offset points", ha="center", color=PAL[1], fontsize=11)
    guardar(fig, f"jer_paso{paso}")

# Cortar el árbol: 2 alturas distintas -> 2 o 3 grupos
for k, corte_y in [(2, (Z[-1, 2] + Z[-2, 2]) / 2), (3, (Z[-2, 2] + Z[-3, 2]) / 2)]:
    lab = fcluster(Z, t=k, criterion="maxclust")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.scatter(toy.x, toy.y, s=140, c=[PAL[l - 1] for l in lab], zorder=3)
    for n, r in toy.iterrows():
        ax.annotate(n, (r.x, r.y), xytext=(7, 5), textcoords="offset points", fontsize=13, weight="bold")
    ax.set_xlim(-0.5, 10.5); ax.set_ylim(-0.8, 6.3)
    ax.set_title(f"Resultado: {k} grupos")
    ax = axes[1]
    for i in range(len(Z)):
        ic, dc = segmentos[i]
        ax.plot(ic, dc, color=GRIS, lw=1.5)
    ax.axhline(corte_y, color="black", ls="--", lw=1.8)
    ax.text(10 * len(toy) - 1, corte_y + 0.25, f"corte → la línea cruza {k} ramas", ha="right")
    ax.set_xticks([xpos[n] for n in orden_hojas], orden_hojas, fontsize=12)
    ax.set_xlim(0, 10 * len(toy)); ax.set_ylim(0, Z[-1, 2] * 1.1)
    ax.set_ylabel("Distancia al unirse"); ax.grid(False)
    ax.set_title(f"Corto el árbol a altura {corte_y:.1f}")
    guardar(fig, f"jer_corte{k}")

paises = pd.read_csv("data/paises_onu.csv").rename(columns={"rownames": "pais"}).dropna()
vars_p = ["fertility", "ppgdp", "lifeExpF", "pctUrban", "infantMortality"]
P = paises[vars_p].copy()
P["ppgdp"] = np.log10(P["ppgdp"])
Ps = StandardScaler().fit_transform(P)

# Todos los países: dendrograma truncado
Zall = linkage(Ps, "ward")
fig, ax = plt.subplots(figsize=(10, 3.8))
dendrogram(Zall, truncate_mode="lastp", p=30, ax=ax, color_threshold=0, above_threshold_color=GRIS,
           show_leaf_counts=True, leaf_font_size=9)
corte = (Zall[-4, 2] + Zall[-3, 2]) / 2
ax.axhline(corte, color=PAL[1], ls="--", lw=1.8)
ax.text(ax.get_xlim()[1], corte + 0.4, "corte → 4 grupos", ha="right", color=PAL[1])
ax.set_ylabel("Distancia (Ward)"); ax.grid(False)
ax.set_title("193 países (se muestran las últimas 30 uniones)")
guardar(fig, "jer_todos")

# América: dendrograma completo con nombres en castellano
ES = {"Argentina": "Argentina", "Aruba": "Aruba", "Bahamas": "Bahamas", "Barbados": "Barbados", "Belize": "Belice",
      "Bolivia": "Bolivia", "Brazil": "Brasil", "Canada": "Canadá", "Chile": "Chile", "Colombia": "Colombia",
      "Costa Rica": "Costa Rica", "Cuba": "Cuba", "Dominican Republic": "Rep. Dominicana", "Ecuador": "Ecuador",
      "El Salvador": "El Salvador", "Grenada": "Granada", "Guatemala": "Guatemala", "Guyana": "Guyana",
      "Haiti": "Haití", "Honduras": "Honduras", "Jamaica": "Jamaica", "Mexico": "México",
      "Neth Antilles": "Antillas Neerl.", "Nicaragua": "Nicaragua", "Panama": "Panamá", "Paraguay": "Paraguay",
      "Peru": "Perú", "Puerto Rico": "Puerto Rico", "Saint Lucia": "Santa Lucía",
      "St Vincent and Grenadines": "San Vicente y Gr.", "Suriname": "Surinam",
      "Trinidad and Tobago": "Trinidad y Tobago", "United States": "Estados Unidos", "Uruguay": "Uruguay",
      "Venezuela": "Venezuela"}
am = paises.region.isin(["Latin Amer", "Caribbean", "North America"]).values
Zam = linkage(Ps[am], "ward")
corte_am = (Zam[-4, 2] + Zam[-3, 2]) / 2
fig, ax = plt.subplots(figsize=(12, 4.6))
from scipy.cluster import hierarchy
hierarchy.set_link_color_palette(PAL[:4])
dendrogram(Zam, labels=[ES[p] for p in paises.pais[am]], ax=ax, color_threshold=corte_am,
           above_threshold_color=GRIS, leaf_rotation=90, leaf_font_size=13)
ax.axhline(corte_am, color="black", ls="--", lw=1.2)
ax.set_ylabel("Distancia (Ward)"); ax.grid(False)
ax.set_title("Países de América según desarrollo (fertilidad, PBI per cápita, esperanza de vida, urbanización, mortalidad infantil)")
guardar(fig, "jer_america")

# ---------------------------------------------------------------------------
# 3) PCA: autos (mapa de posicionamiento)
# ---------------------------------------------------------------------------
autos = pd.read_csv("data/autos_cars93.csv")
NOMBRES = {"Price": "Precio", "MPG.city": "Rend. ciudad", "MPG.highway": "Rend. ruta",
           "EngineSize": "Motor (litros)", "Horsepower": "Potencia (HP)", "RPM": "RPM máx.",
           "Fuel.tank.capacity": "Tanque", "Passengers": "Pasajeros", "Length": "Largo",
           "Wheelbase": "Distancia ejes", "Width": "Ancho", "Weight": "Peso"}
A = autos[list(NOMBRES)].rename(columns=NOMBRES)
As = StandardScaler().fit_transform(A)

# Correlaciones
fig, ax = plt.subplots(figsize=(7.2, 6))
corr = A.corr()
im = ax.imshow(corr, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("vn", ["#b85c2e", "#ffffff", "#1d3b2a"]), vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right")
ax.set_yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        v = corr.iloc[i, j]
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=10, color="white" if abs(v) > 0.6 else "black")
ax.grid(False)
fig.colorbar(im, ax=ax, shrink=0.8, label="Correlación")
ax.set_title("12 variables que dicen casi lo mismo")
guardar(fig, "pca_correlaciones")

# Intuición 2D: largo vs peso
xy = StandardScaler().fit_transform(autos[["Length", "Weight"]])
p2 = PCA(2).fit(xy)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.scatter(xy[:, 0], xy[:, 1], s=26, color=GRIS, alpha=0.7, edgecolor="white", linewidth=0.5)
for v, var, col, n in zip(p2.components_, p2.explained_variance_, [PAL[1], PAL[0]], ["PC1", "PC2"]):
    vec = v * 2 * np.sqrt(var)
    ax.annotate("", xy=vec, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=2.5, color=col))
    ax.text(*(vec * 1.12), n, color=col, weight="bold", fontsize=13)
ax.set_aspect("equal"); ax.set_ylim(-2.8, 2.6); ax.set_xlabel("Largo (estandarizado)"); ax.set_ylabel("Peso (estandarizado)")
ax.set_title("PC1 = la dirección donde más se estiran los datos", pad=14)
proj = p2.transform(xy)
ax = axes[1]
ax.scatter(proj[:, 0], np.zeros(len(proj)), s=26, color=PAL[1], alpha=0.6)
ax.set_yticks([]); ax.set_ylim(-1, 1); ax.grid(False)
ax.spines["left"].set_visible(False)
ax.set_xlabel("PC1 ≈ \"tamaño\" del auto")
ax.set_title(f"Un solo número conserva el {p2.explained_variance_ratio_[0]:.0%} de la info")
ax.annotate("chicos", (proj[:, 0].min(), 0.15), color=GRIS)
ax.annotate("grandes", (proj[:, 0].max(), 0.15), ha="right", color=GRIS)
guardar(fig, "pca_idea")

pca = PCA().fit(As)
ev = pca.explained_variance_ratio_
fig, ax = plt.subplots(figsize=(7.5, 3.8))
comp = np.arange(1, len(ev) + 1)
ax.bar(comp, ev * 100, color=PAL[0], width=0.7, label="Cada componente")
ax.plot(comp, ev.cumsum() * 100, marker="o", color=PAL[1], lw=2, label="Acumulado")
for i in range(3):
    ax.text(comp[i], ev.cumsum()[i] * 100 + 3, f"{ev.cumsum()[i]:.0%}", ha="center", color=GRIS)
ax.set_xticks(comp, [f"PC{i}" for i in comp], fontsize=9)
ax.set_ylabel("% de la variabilidad"); ax.set_ylim(0, 110)
ax.legend(loc="center right")
ax.set_title("Con 2 componentes ya resumimos ~78% de 12 variables")
guardar(fig, "pca_varianza")

load = pd.DataFrame(pca.components_[:2].T, index=A.columns, columns=["PC1", "PC2"])
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False, gridspec_kw={"wspace": 0.55})
for ax, pc, tit in zip(axes, ["PC1", "PC2"], ["PC1 = TAMAÑO (grande y gastador ↔ chico y económico)",
                                               "PC2 = DEPORTIVO/PREMIUM ↔ FAMILIAR"]):
    s = load[pc].sort_values()
    ax.barh(s.index, s.values, color=[PAL[1] if v < 0 else PAL[0] for v in s.values], height=0.7)
    ax.axvline(0, color=GRIS, lw=1)
    ax.set_title(tit, fontsize=11)
    ax.set_xlabel("Peso de la variable (loading)")
guardar(fig, "pca_loadings")

Z2 = pca.transform(As)[:, :2]
tipos = {"Small": "Chico", "Compact": "Compacto", "Midsize": "Mediano", "Large": "Grande",
         "Sporty": "Deportivo", "Van": "Utilitario"}
fig, ax = plt.subplots(figsize=(9, 5.6))
for i, (t, ts) in enumerate(tipos.items()):
    m = autos.Type == t
    ax.scatter(Z2[m, 0], Z2[m, 1], s=40, color=PAL[i], edgecolor="white", linewidth=0.6, label=ts)
destacados = ["Chevrolet Corvette", "Mazda RX-7", "Geo Metro", "Ford Festiva", "Chevrolet Astro",
              "Lincoln Town_Car", "Mercedes-Benz 300E", "Honda Civic", "Ford Aerostar", "Dodge Stealth",
              "Cadillac DeVille", "Hyundai Excel"]
for nombre in destacados:
    i = autos.index[autos.Make == nombre]
    if len(i):
        i = i[0]
        ax.annotate(nombre.replace("_", " "), Z2[i], xytext=(5, 4), textcoords="offset points", fontsize=11, color=TINTA)
ax.axhline(0, color="#cfc8b6", lw=1); ax.axvline(0, color="#cfc8b6", lw=1)
ax.set_xlabel(f"PC1: tamaño  ({ev[0]:.0%})   chico/económico  ←→  grande/gastador")
ax.set_ylabel(f"PC2 ({ev[1]:.0%})   familiar  ←→  deportivo/premium")
ax.legend(title="Tipo", loc="upper left", bbox_to_anchor=(1.0, 1.0))
ax.set_title("Mapa de posicionamiento de 93 modelos de autos")
guardar(fig, "pca_mapa")

print("Figuras generadas en img/")
