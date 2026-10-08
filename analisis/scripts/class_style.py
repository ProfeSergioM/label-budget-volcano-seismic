"""
Estilo visual compartido por clase (LP, VT, HB, LV/VLP, AV, TO), para que la
Figura 2 (galeria de formas de onda) y la Figura 3 (series temporales) usen
exactamente los mismos colores -- una sola fuente de verdad en vez de dos
paletas definidas por separado que puedan divergir.
"""
from __future__ import annotations

CLASS_ORDER = ["LP", "VT", "HB", "LV", "AV", "TO"]

# LV es el codigo de base de datos para VLP (very long-period) -- ver
# data_section.tex.
CLASS_DISPLAY = {"LP": "LP", "VT": "VT", "HB": "HB", "LV": "VLP", "AV": "AV", "TO": "TO"}

CLASS_COLOR = {
    "LP": "#1a4d7a",
    "VT": "#c0392b",
    "HB": "#7d3c98",
    "LV": "#16a085",
    "AV": "#d68910",
    "TO": "#4a4a4a",
}

# Estilo de linea y marcador distintos por clase (redundancia deliberada con
# el color: distinguibles tambien impresos en blanco y negro).
CLASS_LINESTYLE = {
    "LP": "-",
    "VT": "--",
    "HB": "-.",
    "LV": ":",
    "AV": (0, (5, 1)),
    "TO": (0, (1, 1)),
}

CLASS_MARKER = {
    "LP": "o",
    "VT": "s",
    "HB": "^",
    "LV": "D",
    "AV": "v",
    "TO": "*",
}


# Sombreado alterno de fila, compartido por las figuras que apilan paneles
# (galeria de formas de onda y ventana de analisis). Se aplica solo sobre los
# margenes alrededor de cada panel, nunca sobre el fondo del panel, que queda
# blanco.
ROW_SHADE_COLOR = "#e8edf2"
