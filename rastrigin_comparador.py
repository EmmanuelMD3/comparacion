"""Comparador de Algoritmo Genético (AG) y Enjambre de Partículas (PSO).

Actividad: minimizar la función de Rastrigin en [-5.12, 5.12]^n.

Ejecutar con interfaz:
    python rastrigin_comparador.py

Ejecutar sin interfaz (terminal):
    python rastrigin_comparador.py --cli --dimensiones 2 --iteraciones 200 --corridas 10

Dependencias:
    pip install numpy matplotlib
"""

from __future__ import annotations

import argparse
import csv
import queue
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

A = 10.0
LIMITE_INF = -5.12
LIMITE_SUP = 5.12
TOLERANCIA = 0.01


def rastrigin(x: np.ndarray) -> np.ndarray | float:
    """Función Rastrigin. Acepta un vector (n,) o población (m, n)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return float(A * x.size + np.sum(x**2 - A * np.cos(2 * np.pi * x)))
    if x.ndim == 2:
        return A * x.shape[1] + np.sum(x**2 - A * np.cos(2 * np.pi * x), axis=1)
    raise ValueError("x debe ser un vector o una matriz de individuos")


@dataclass(frozen=True)
class Config:
    dimensiones: int = 2
    poblacion: int = 50
    iteraciones: int = 200
    corridas: int = 10
    semilla: int = 42

    def validar(self) -> None:
        if not (1 <= self.dimensiones <= 30):
            raise ValueError("Las dimensiones deben estar entre 1 y 30")
        if not (10 <= self.poblacion <= 500):
            raise ValueError("La población debe estar entre 10 y 500")
        if not (1 <= self.iteraciones <= 2000):
            raise ValueError("Las iteraciones deben estar entre 1 y 2000")
        if not (1 <= self.corridas <= 100):
            raise ValueError("Las corridas deben estar entre 1 y 100")
        if not (0 <= self.semilla <= 2_147_483_647):
            raise ValueError("La semilla debe ser un entero no negativo válido")


@dataclass
class Resultado:
    valor: float
    posicion: np.ndarray
    historia: np.ndarray
    segundos: float
    evaluaciones: int


def algoritmo_genetico(config: Config, semilla: int) -> Resultado:
    """AG real: selección por torneo, cruce BLX, mutación gaussiana y elitismo."""
    rng = np.random.default_rng(semilla)
    t0 = time.perf_counter()
    p = config.poblacion
    n = config.dimensiones
    poblacion = rng.uniform(LIMITE_INF, LIMITE_SUP, size=(p, n))
    aptitudes = rastrigin(poblacion)
    idx = int(np.argmin(aptitudes))
    mejor_valor = float(aptitudes[idx])
    mejor_posicion = poblacion[idx].copy()
    historia = [mejor_valor]
    num_elites = max(1, int(p * 0.05))
    prob_mutacion = max(0.08, min(0.25, 1.0 / n))

    def torneo() -> np.ndarray:
        candidatos = rng.integers(0, p, size=3)
        elegido = candidatos[int(np.argmin(aptitudes[candidatos]))]
        return poblacion[elegido]

    for i in range(config.iteraciones):
        orden = np.argsort(aptitudes)
        siguiente = np.empty_like(poblacion)
        siguiente[:num_elites] = poblacion[orden[:num_elites]]
        progreso = (i + 1) / config.iteraciones
        desviacion_mutacion = (LIMITE_SUP - LIMITE_INF) * (0.12 * (1 - progreso) + 0.002)

        for j in range(num_elites, p):
            padre1, padre2 = torneo(), torneo()
            if rng.random() < 0.90:
                mezcla = rng.uniform(-0.25, 1.25, size=n)
                hijo = padre1 + mezcla * (padre2 - padre1)
            else:
                hijo = padre1.copy()

            mascara = rng.random(n) < prob_mutacion
            if np.any(mascara):
                hijo[mascara] += rng.normal(0, desviacion_mutacion, size=np.sum(mascara))
            siguiente[j] = np.clip(hijo, LIMITE_INF, LIMITE_SUP)

        poblacion = siguiente
        aptitudes = rastrigin(poblacion)
        idx = int(np.argmin(aptitudes))
        if aptitudes[idx] < mejor_valor:
            mejor_valor = float(aptitudes[idx])
            mejor_posicion = poblacion[idx].copy()
        historia.append(mejor_valor)

    return Resultado(
        valor=mejor_valor,
        posicion=mejor_posicion,
        historia=np.array(historia),
        segundos=time.perf_counter() - t0,
        evaluaciones=p * (config.iteraciones + 1),
    )


def enjambre_particulas(config: Config, semilla: int) -> Resultado:
    """PSO global-best con inercia decreciente y velocidad limitada."""
    rng = np.random.default_rng(semilla)
    t0 = time.perf_counter()
    p, n = config.poblacion, config.dimensiones
    amplitud = LIMITE_SUP - LIMITE_INF
    vmax = 0.2 * amplitud
    posiciones = rng.uniform(LIMITE_INF, LIMITE_SUP, size=(p, n))
    velocidades = rng.uniform(-vmax, vmax, size=(p, n))
    personales = posiciones.copy()
    personales_valores = rastrigin(posiciones)
    indice_mejor = int(np.argmin(personales_valores))
    mejor_posicion = personales[indice_mejor].copy()
    mejor_valor = float(personales_valores[indice_mejor])
    historia = [mejor_valor]

    for i in range(config.iteraciones):
        progreso = i / max(1, config.iteraciones - 1)
        inercia = 0.90 - 0.50 * progreso
        aleatorio1 = rng.random((p, n))
        aleatorio2 = rng.random((p, n))
        velocidades = (
            inercia * velocidades
            + 1.7 * aleatorio1 * (personales - posiciones)
            + 1.7 * aleatorio2 * (mejor_posicion - posiciones)
        )
        velocidades = np.clip(velocidades, -vmax, vmax)
        nuevas_posiciones = posiciones + velocidades
        fuera = (nuevas_posiciones < LIMITE_INF) | (nuevas_posiciones > LIMITE_SUP)
        posiciones = np.clip(nuevas_posiciones, LIMITE_INF, LIMITE_SUP)
        velocidades[fuera] = 0.0

        valores = rastrigin(posiciones)
        mejoraron = valores < personales_valores
        personales[mejoraron] = posiciones[mejoraron]
        personales_valores[mejoraron] = valores[mejoraron]
        indice_mejor = int(np.argmin(personales_valores))
        if personales_valores[indice_mejor] < mejor_valor:
            mejor_valor = float(personales_valores[indice_mejor])
            mejor_posicion = personales[indice_mejor].copy()
        historia.append(mejor_valor)

    return Resultado(
        valor=mejor_valor,
        posicion=mejor_posicion,
        historia=np.array(historia),
        segundos=time.perf_counter() - t0,
        evaluaciones=p * (config.iteraciones + 1),
    )


def ejecutar_experimento(
    config: Config,
    progreso: Callable[[int, int], None] | None = None,
) -> dict[str, list[Resultado]]:
    config.validar()
    resultados: dict[str, list[Resultado]] = {"AG": [], "PSO": []}
    total = config.corridas * 2
    completados = 0
    for corrida in range(config.corridas):
        semilla = config.semilla + corrida
        resultados["AG"].append(algoritmo_genetico(config, semilla))
        completados += 1
        if progreso:
            progreso(completados, total)
        resultados["PSO"].append(enjambre_particulas(config, semilla))
        completados += 1
        if progreso:
            progreso(completados, total)
    return resultados


def resumen(config: Config, resultados: dict[str, list[Resultado]]) -> str:
    salida = [
        "COMPARACIÓN DE OPTIMIZADORES — FUNCIÓN RASTRIGIN",
        f"f(x) = 10n + Σ[xᵢ² − 10 cos(2πxᵢ)]    n = {config.dimensiones}",
        "Dominio: [-5.12, 5.12] por variable; mínimo teórico: f(0) = 0",
        f"Condiciones: {config.poblacion} individuos/partículas, {config.iteraciones} iteraciones, "
        f"{config.corridas} corridas, semilla inicial {config.semilla}",
        f"Presupuesto idéntico: {config.poblacion * (config.iteraciones + 1):,} "
        "evaluaciones de f(x) por corrida y algoritmo",
        f"Éxito: encontrar f(x) ≤ {TOLERANCIA}",
        "",
    ]
    for nombre in ("AG", "PSO"):
        corridas = resultados[nombre]
        valores = np.array([r.valor for r in corridas])
        tiempos = np.array([r.segundos for r in corridas])
        mejor = min(corridas, key=lambda r: r.valor)
        salida.extend([
            f"{nombre} — {'Algoritmo genético' if nombre == 'AG' else 'Enjambre de partículas'}",
            f"  Mejor resultado:        {np.min(valores):.10f}",
            f"  Promedio del resultado: {np.mean(valores):.10f}",
            f"  Mediana del resultado:  {np.median(valores):.10f}",
            f"  Desv. estándar:         {np.std(valores):.10f}",
            f"  Tiempo promedio:        {np.mean(tiempos) * 1000:.2f} ms",
            f"  Tasa de éxito:          {np.mean(valores <= TOLERANCIA) * 100:.1f}%",
            f"  Mejor posición:         {np.array2string(mejor.posicion, precision=5, suppress_small=True)}",
            "",
        ])
    media_ag = float(np.mean([r.valor for r in resultados["AG"]]))
    media_pso = float(np.mean([r.valor for r in resultados["PSO"]]))
    diferencia = abs(media_ag - media_pso)
    if diferencia <= 1e-8:
        ganador = "Empate práctico (diferencia menor o igual a 1e-8)"
    else:
        ganador = "AG" if media_ag < media_pso else "PSO"
    salida.append(f"MEJOR RESULTADO PROMEDIO (menor es mejor): {ganador}")
    salida.append("Nota: el tiempo depende del equipo; el algoritmo más rápido no siempre encuentra el mejor mínimo.")
    return "\n".join(salida)


def guardar_csv(ruta: str, config: Config, resultados: dict[str, list[Resultado]]) -> tuple[Path, Path]:
    base = Path(ruta)
    if base.suffix.lower() == ".csv":
        base = base.with_suffix("")
    archivo_corridas = base.with_name(base.name + "_resultados.csv")
    archivo_convergencia = base.with_name(base.name + "_convergencia.csv")
    with archivo_corridas.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["algoritmo", "corrida", "dimensiones", "poblacion", "iteraciones", "semilla",
                         "mejor_fitness", "tiempo_segundos", "evaluaciones", "exito", "mejor_posicion"])
        for metodo, lista in resultados.items():
            for i, res in enumerate(lista):
                writer.writerow([metodo, i + 1, config.dimensiones, config.poblacion,
                                 config.iteraciones, config.semilla + i, f"{res.valor:.12g}",
                                 f"{res.segundos:.8f}", res.evaluaciones,
                                 int(res.valor <= TOLERANCIA),
                                 " ".join(f"{v:.8g}" for v in res.posicion)])
    with archivo_convergencia.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["iteracion", "ag_mejor_promedio", "pso_mejor_promedio"])
        medias = {m: np.mean(np.stack([r.historia for r in lista]), axis=0)
                  for m, lista in resultados.items()}
        for i in range(config.iteraciones + 1):
            writer.writerow([i, f"{medias['AG'][i]:.12g}", f"{medias['PSO'][i]:.12g}"])
    return archivo_corridas, archivo_convergencia


def graficar_convergencia(fig, config: Config, resultados: dict[str, list[Resultado]]) -> None:
    fig.clear()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    nombres = ("AG", "PSO")
    colores = {"AG": "#0F766E", "PSO": "#D97706"}
    for nombre in nombres:
        historias = np.stack([r.historia for r in resultados[nombre]])
        promedio = np.mean(historias, axis=0)
        # La escala logarítmica permite apreciar mejoras próximas a cero.
        ax1.plot(np.arange(len(promedio)), np.maximum(promedio, 1e-12),
                 label=nombre, linewidth=2, color=colores[nombre])
    ax1.axhline(TOLERANCIA, linestyle="--", color="gray", linewidth=1,
                label=f"Tolerancia {TOLERANCIA}")
    ax1.set_yscale("log")
    ax1.set_xlabel("Iteración")
    ax1.set_ylabel("Mejor fitness promedio (escala log)")
    ax1.set_title("Convergencia promedio")
    ax1.grid(alpha=0.25)
    ax1.legend()

    data = [[r.valor for r in resultados[n]] for n in nombres]
    bp = ax2.boxplot(data, tick_labels=list(nombres), patch_artist=True,
                     showmeans=True, meanline=False, widths=0.45)
    for patch, nombre in zip(bp["boxes"], nombres):
        patch.set_facecolor(colores[nombre])
        patch.set_alpha(0.45)
    ax2.set_title("Resultados finales por corrida")
    ax2.set_ylabel("Mejor fitness alcanzado")
    ax2.grid(alpha=0.2, axis="y")
    fig.suptitle(f"Función Rastrigin • {config.dimensiones} dimensiones • {config.corridas} corridas",
                 fontweight="bold", fontsize=12)
    fig.subplots_adjust(top=0.83, bottom=0.18, left=0.12, right=0.96, wspace=0.36)


def graficar_superficie(fig, config: Config, resultados: dict[str, list[Resultado]]) -> None:
    fig.clear()
    ax = fig.add_subplot(111)
    if config.dimensiones != 2:
        ax.text(0.5, 0.5, "El mapa de contorno solo está disponible\npara 2 dimensiones.",
                horizontalalignment="center", verticalalignment="center",
                transform=ax.transAxes, fontsize=13)
        ax.set_axis_off()
        return
    coords = np.linspace(LIMITE_INF, LIMITE_SUP, 260)
    xx, yy = np.meshgrid(coords, coords)
    zz = 20 + (xx * xx - 10 * np.cos(2 * np.pi * xx)) + (yy * yy - 10 * np.cos(2 * np.pi * yy))
    contorno = ax.contourf(xx, yy, zz, levels=35, cmap="viridis")
    fig.colorbar(contorno, ax=ax, label="f(x₁, x₂)", shrink=0.87)
    for nombre, color, marcador in [("AG", "#EF4444", "o"), ("PSO", "#FACC15", "X")]:
        mejor = min(resultados[nombre], key=lambda r: r.valor)
        ax.scatter(mejor.posicion[0], mejor.posicion[1], color=color,
                   marker=marcador, s=125, edgecolors="black", linewidths=0.8,
                   label=f"{nombre}: f={mejor.valor:.4g}", zorder=4)
    ax.scatter(0, 0, marker="*", s=190, color="white", edgecolors="black",
               linewidths=0.8, label="Óptimo (0, 0)", zorder=5)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title("Región de búsqueda y mejores soluciones")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=9)
    fig.subplots_adjust(left=0.13, right=0.85, bottom=0.12, top=0.92)


class Aplicacion:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        from tkinter.scrolledtext import ScrolledText
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.root.title("Comparador de Rastrigin | AG vs PSO")
        self.root.geometry("1240x810")
        self.root.minsize(1050, 700)
        self.root.configure(bg="#eef3f7")
        self.cola: queue.Queue = queue.Queue()
        self.resultados: dict[str, list[Resultado]] | None = None
        self.config_ultima: Config | None = None
        self.trabajando = False

        estilo = ttk.Style(root)
        if "clam" in estilo.theme_names():
            estilo.theme_use("clam")
        estilo.configure("Principal.TFrame", background="#eef3f7")
        estilo.configure("Panel.TFrame", background="white")
        estilo.configure("Panel.TLabel", background="white", foreground="#14253b")
        estilo.configure("Titulo.TLabel", background="#eef3f7", foreground="#123456",
                         font=("Segoe UI", 19, "bold"))
        estilo.configure("Subtitulo.TLabel", background="#eef3f7", foreground="#50657b",
                         font=("Segoe UI", 10))
        estilo.configure("TLabelframe", background="white")
        estilo.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"),
                         background="white", foreground="#123456")
        estilo.configure("TButton", padding=(9, 8), font=("Segoe UI", 9))
        estilo.configure("Accion.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 10))

        titulo = ttk.Frame(root, style="Principal.TFrame", padding=(20, 12, 20, 7))
        titulo.pack(fill="x")
        ttk.Label(titulo, text="Optimización de Rastrigin", style="Titulo.TLabel").pack(anchor="w")
        ttk.Label(titulo, text="Comparación experimental: Algoritmo Genético (AG) vs. Particle Swarm Optimization (PSO)",
                  style="Subtitulo.TLabel").pack(anchor="w", pady=(3, 0))

        cuerpo = ttk.Frame(root, style="Principal.TFrame", padding=(16, 8, 16, 12))
        cuerpo.pack(fill="both", expand=True)
        cuerpo.columnconfigure(0, weight=0)
        cuerpo.columnconfigure(1, weight=1)
        cuerpo.rowconfigure(0, weight=1)

        panel = ttk.Frame(cuerpo, style="Panel.TFrame", padding=18, width=285)
        panel.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        panel.grid_propagate(False)

        marco = ttk.LabelFrame(panel, text="Parámetros de la prueba", padding=12)
        marco.pack(fill="x", pady=(0, 15))
        campos = [
            ("Dimensiones (n)", "dimensiones", "2"),
            ("Individuos / partículas", "poblacion", "50"),
            ("Iteraciones", "iteraciones", "200"),
            ("Corridas independientes", "corridas", "10"),
            ("Semilla aleatoria", "semilla", "42"),
        ]
        self.variables = {}
        for i, (etiqueta, clave, inicial) in enumerate(campos):
            ttk.Label(marco, text=etiqueta, background="white").grid(row=2 * i, column=0, sticky="w", pady=(7, 2))
            var = tk.StringVar(value=inicial)
            self.variables[clave] = var
            ttk.Entry(marco, textvariable=var, width=24).grid(row=2 * i + 1, column=0, sticky="ew")
        marco.columnconfigure(0, weight=1)

        info = ttk.LabelFrame(panel, text="Problema a resolver", padding=12)
        info.pack(fill="x", pady=(0, 12))
        for linea in [
            "f(x) = 10n + Σ[xᵢ² − 10 cos(2πxᵢ)]",
            "A = 10",
            "−5.12 ≤ xᵢ ≤ 5.12",
            "Mínimo global: f(0, ..., 0) = 0",
            "Éxito si f(x) ≤ 0.01",
        ]:
            ttk.Label(info, text=linea, style="Panel.TLabel",
                      wraplength=240).pack(anchor="w", pady=3)

        self.boton_ejecutar = ttk.Button(panel, text="▶  Ejecutar comparación",
                                         style="Accion.TButton", command=self.ejecutar)
        self.boton_ejecutar.pack(fill="x", pady=(4, 10))
        self.boton_csv = ttk.Button(panel, text="Exportar datos CSV", command=self.exportar_csv, state="disabled")
        self.boton_csv.pack(fill="x", pady=(0, 8))
        self.boton_imagen = ttk.Button(panel, text="Guardar gráfica PNG", command=self.guardar_grafica, state="disabled")
        self.boton_imagen.pack(fill="x")
        self.estado = ttk.Label(panel, text="Listo para ejecutar", style="Panel.TLabel", wraplength=242)
        self.estado.pack(fill="x", pady=(18, 7))
        self.progreso = ttk.Progressbar(panel, mode="determinate", maximum=100)
        self.progreso.pack(fill="x")

        derecha = ttk.Frame(cuerpo, style="Panel.TFrame", padding=12)
        derecha.grid(row=0, column=1, sticky="nsew")
        derecha.rowconfigure(0, weight=3)
        derecha.rowconfigure(1, weight=2)
        derecha.columnconfigure(0, weight=1)

        self.tabs = ttk.Notebook(derecha)
        self.tabs.grid(row=0, column=0, sticky="nsew")
        tab_convergencia = ttk.Frame(self.tabs)
        tab_mapa = ttk.Frame(self.tabs)
        self.tabs.add(tab_convergencia, text="Convergencia y resultados")
        self.tabs.add(tab_mapa, text="Mapa de contorno (2D)")
        self.figuras = [Figure(figsize=(8.2, 4.5), dpi=100, facecolor="white"),
                        Figure(figsize=(8.2, 4.5), dpi=100, facecolor="white")]
        self.canvases = []
        for tab, figura in zip([tab_convergencia, tab_mapa], self.figuras):
            canvas = FigureCanvasTkAgg(figura, master=tab)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            self.canvases.append(canvas)
        for figura, mensaje in zip(self.figuras, [
            "Ejecuta una comparación para ver las gráficas de convergencia",
            "Aquí aparecerá el mapa de contorno para n = 2",
        ]):
            ax = figura.add_subplot(111)
            ax.text(0.5, 0.5, mensaje, ha="center", va="center", fontsize=12,
                    transform=ax.transAxes, color="#64748b")
            ax.set_axis_off()
        for canvas in self.canvases:
            canvas.draw_idle()

        resultados_frame = ttk.LabelFrame(derecha, text="Análisis de resultados", padding=7)
        resultados_frame.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        self.texto = ScrolledText(resultados_frame, wrap="word", font=("Consolas", 9),
                                 height=12, bg="#fafcfe", fg="#14253b", relief="flat")
        self.texto.pack(fill="both", expand=True)
        self.escribir("Configura los parámetros y pulsa «Ejecutar comparación».\n"
                      "Se calcularán múltiples corridas para comparar calidad, tiempo y éxito.")

    def escribir(self, texto: str):
        self.texto.configure(state="normal")
        self.texto.delete("1.0", "end")
        self.texto.insert("1.0", texto)
        self.texto.configure(state="disabled")

    def ejecutar(self):
        from tkinter import messagebox
        try:
            config = Config(**{clave: int(var.get()) for clave, var in self.variables.items()})
            config.validar()
        except ValueError as exc:
            messagebox.showerror("Parámetros incorrectos", str(exc))
            return
        self.trabajando = True
        self.resultados = None
        self.config_ultima = None
        self.boton_ejecutar.configure(state="disabled")
        self.boton_csv.configure(state="disabled")
        self.boton_imagen.configure(state="disabled")
        self.progreso["value"] = 0
        self.estado.configure(text="Ejecutando pruebas...")
        self.escribir("Se están ejecutando AG y PSO con el mismo presupuesto de evaluaciones...")

        def tarea():
            try:
                datos = ejecutar_experimento(
                    config,
                    progreso=lambda n, total: self.cola.put(("avance", n, total)),
                )
                self.cola.put(("exito", config, datos))
            except Exception as exc:
                self.cola.put(("error", str(exc)))

        threading.Thread(target=tarea, daemon=True).start()
        self.root.after(100, self.revisar_cola)

    def revisar_cola(self):
        from tkinter import messagebox
        try:
            while True:
                item = self.cola.get_nowait()
                if item[0] == "avance":
                    _, actual, total = item
                    self.progreso["value"] = actual / total * 100
                    self.estado.configure(text=f"Corridas ejecutadas: {actual}/{total}")
                elif item[0] == "exito":
                    _, config, datos = item
                    self.config_ultima = config
                    self.resultados = datos
                    graficar_convergencia(self.figuras[0], config, datos)
                    graficar_superficie(self.figuras[1], config, datos)
                    for canvas in self.canvases:
                        canvas.draw_idle()
                    self.escribir(resumen(config, datos))
                    self.estado.configure(text="Comparación terminada correctamente")
                    self.progreso["value"] = 100
                    self.boton_ejecutar.configure(state="normal")
                    self.boton_csv.configure(state="normal")
                    self.boton_imagen.configure(state="normal")
                    self.trabajando = False
                elif item[0] == "error":
                    self.boton_ejecutar.configure(state="normal")
                    self.estado.configure(text="Error en el experimento")
                    self.trabajando = False
                    messagebox.showerror("Error", item[1])
        except queue.Empty:
            pass
        if self.trabajando:
            self.root.after(100, self.revisar_cola)

    def exportar_csv(self):
        from tkinter import filedialog, messagebox
        if self.resultados is None or self.config_ultima is None:
            return
        ruta = filedialog.asksaveasfilename(title="Nombre base para los CSV",
                                            defaultextension=".csv", filetypes=[("CSV", "*.csv")],
                                            initialfile="comparacion_rastrigin.csv")
        if ruta:
            try:
                uno, dos = guardar_csv(ruta, self.config_ultima, self.resultados)
                messagebox.showinfo("Datos exportados", f"Archivos guardados:\n{uno}\n{dos}")
            except OSError as exc:
                messagebox.showerror("Error al guardar", str(exc))

    def guardar_grafica(self):
        from tkinter import filedialog, messagebox
        if self.resultados is None:
            return
        ruta = filedialog.asksaveasfilename(title="Guardar gráfica actual", defaultextension=".png",
                                            filetypes=[("Imagen PNG", "*.png")],
                                            initialfile="rastrigin_grafica.png")
        if ruta:
            try:
                self.figuras[self.tabs.index("current")].savefig(ruta, dpi=160)
                messagebox.showinfo("Gráfica guardada", f"Se guardó la imagen en:\n{ruta}")
            except OSError as exc:
                messagebox.showerror("Error al guardar", str(exc))


def main():
    parser = argparse.ArgumentParser(description="Comparar AG y PSO para minimizar Rastrigin")
    parser.add_argument("--cli", action="store_true", help="Ejecutar en terminal, sin interfaz gráfica")
    parser.add_argument("--dimensiones", type=int, default=2)
    parser.add_argument("--poblacion", type=int, default=50)
    parser.add_argument("--iteraciones", type=int, default=200)
    parser.add_argument("--corridas", type=int, default=10)
    parser.add_argument("--semilla", type=int, default=42)
    parser.add_argument("--salida", type=str, default=None, help="Ruta base para exportar CSV en modo CLI")
    argumentos = parser.parse_args()
    config = Config(argumentos.dimensiones, argumentos.poblacion,
                    argumentos.iteraciones, argumentos.corridas, argumentos.semilla)
    if argumentos.cli:
        datos = ejecutar_experimento(config)
        print(resumen(config, datos))
        if argumentos.salida:
            rutas = guardar_csv(argumentos.salida, config, datos)
            print("\nArchivos guardados:", *rutas, sep="\n")
    else:
        import tkinter as tk
        root = tk.Tk()
        Aplicacion(root)
        root.mainloop()


if __name__ == "__main__":
    main()
