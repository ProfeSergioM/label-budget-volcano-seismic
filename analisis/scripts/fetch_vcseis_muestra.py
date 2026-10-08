"""
Baja de VCSEIS solo las trazas que entran a la cola de corte manual.

Los hdf5 de Alaska y Hawai pesan 43 GB entre los cuatro y se borraron despues
de la extraccion (vcseis_nocturno.sh); el disco no tiene espacio para
volverlos a bajar enteros. El servidor de SeisBench acepta lecturas por rango,
asi que h5py lee el archivo remoto a traves de un objeto que pide por HTTP
solo los bloques que toca, y cada traza se guarda en un hdf5 local chico con
la misma ruta interna (data/<bucket>) e indice.

La muestra es la misma que arma cortar_trazas.py: VCSEIS_N trazas por trozo,
con la misma semilla y los mismos filtros, de modo que la cola la encuentra
sin cambios.

Uso:
    python scripts/fetch_vcseis_muestra.py --chunks ak_lp ak_rg hw12t21_lp hw12t21_rg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from cortar_trazas import muestra_vcseis, muestra_vcseis_otras  # noqa: E402
from extract_vcseis import parse_trace_name  # noqa: E402

URL = "https://hifis-storage.desy.de/Helmholtz/HelmholtzAI/SeisBench/datasets/vcseis"
BASE = ROOT / "data" / "external" / "vcseis"
BLOQUE = 1 << 20


class ArchivoHTTP:
    """Archivo de solo lectura sobre HTTP Range, con cache de bloques."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        self.s = requests.Session()
        self.size = int(self.s.head(url, timeout=60).headers["Content-Length"])
        self.cache: dict[int, bytes] = {}

    def _bloque(self, k: int) -> bytes:
        if k not in self.cache:
            a = k * BLOQUE
            b = min(a + BLOQUE, self.size) - 1
            for intento in range(5):
                try:
                    r = self.s.get(self.url, headers={"Range": f"bytes={a}-{b}"},
                                   timeout=120)
                    r.raise_for_status()
                    break
                except requests.RequestException:
                    if intento == 4:
                        raise
            if len(self.cache) > 256:
                self.cache.clear()
            self.cache[k] = r.content
        return self.cache[k]

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out, p = bytearray(), self.pos
        while len(out) < n:
            k, off = divmod(p + len(out), BLOQUE)
            out += self._bloque(k)[off:off + n - len(out)]
        self.pos += n
        return bytes(out)

    def readinto(self, b) -> int:
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def tell(self) -> int:
        return self.pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True


def bajar(chunk: str, n: int, semilla: int, otras: bool = False):
    import h5py
    dest = BASE / f"muestra_{chunk}.hdf5"
    m = (muestra_vcseis_otras(chunk, 500, semilla) if otras
         else muestra_vcseis(chunk, n, semilla))
    remoto = h5py.File(ArchivoHTTP(f"{URL}/waveforms_{chunk}.hdf5"), "r")
    hechas = 0
    with h5py.File(dest, "a") as out:
        for k, name in enumerate(m.trace_name, 1):
            b, i = parse_trace_name(name)
            ruta = f"data/{b}/{i}"
            if ruta not in out:
                largo = int(name.rsplit(":", 1)[1])        # sin el relleno del bucket
                out[ruta] = np.asarray(remoto["data"][b][i, :, :largo], dtype=np.float32)
                hechas += 1
            if k % 50 == 0:
                print(f"  {chunk}: {k}/{len(m)}", flush=True)
    print(f"{chunk}: {hechas} trazas nuevas -> {dest}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", nargs="+",
                    default=["ak_lp", "ak_rg", "hw12t21_lp", "hw12t21_rg"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--semilla", type=int, default=41)
    ap.add_argument("--otras", action="store_true",
                    help="las clases de VCSEIS_OTRAS (hasta 500 por clase) en vez de LP/VT")
    a = ap.parse_args()
    for c in a.chunks:
        bajar(c, a.n, a.semilla, a.otras)


if __name__ == "__main__":
    main()
