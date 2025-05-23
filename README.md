# Proyecto Progra Avanzada - Modelado Bayesiano en Astronomía con PyMC y Numpyro

Este proyecto implementa un modelo bayesiano para estudiar la relación entre la masa de agujeros negros supermasivos y la dispersión de velocidad de las estrellas en el bulbo de galaxias, una problemática clásica en la astronomía moderna.

---

## Contexto Astronómico

En investigación astronómica es común enfrentar el desafío de mediciones con errores significativos. Este proyecto aborda el análisis de la conocida **relación M–σ (masa-disperción)**, donde se estudia la masa del agujero negro supermasivo central (en unidades solares) en función de la dispersión de velocidad de las estrellas (σ, en km/s).

Esta relación fue popularizada en trabajos como Djorgovski y Davis (1987), Merritt (2000) y actualizada en H2013 (Harris et al., 2013), siendo fundamental para la estimación de masas de agujeros negros en galaxias distantes.

El modelo clásico (Tremaine et al., 2002) se expresa como:

$$
\log \left(\frac{M_{\text{blackhole}}}{M_{\odot}}\right) = \alpha + \beta \cdot \log \left(\frac{\sigma}{\sigma_0}\right)
$$

donde:

- M_blackhole es la masa del agujero negro supermasivo,
- M_sun es la masa solar,
- σ es la dispersión de velocidad estelar,
- σ₀ = 200 km/s es un valor de referencia fijo,
- α, β son parámetros lineales a estimar.

---

## Modelo Bayesiano

Este proyecto implementa un modelo bayesiano gaussiano que incorpora explícitamente los errores en las mediciones de ambas variables, usando los frameworks PyMC y Numpyro. Se modela la dispersión extra (scatter) alrededor de la relación lineal mediante el parámetro \( \epsilon \).

---

## Archivos y Estructura

- `blackhole.py`: Modelo bayesiano con PyMC.
- `main.py`: Runner del modelo con dash incorporado.
- `requirements.txt`: Dependencias para instalar.
- `README.md`: Documentación y guía.

---

## Instalación

Requiere Python 3.8+.

Instalar dependencias:

```bash
pip install -r requirements.txt