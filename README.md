# QuantLab ML V3

Panel académico para explorar precios históricos de futuros, escenarios de mercado, resultados de especialistas y carteras simuladas.

## Instalar y ejecutar

Descarga el ZIP del repositorio y extráelo, o clona el proyecto con Git. Desde la carpeta donde se encuentra `app.py`, con Python 3.12 instalado, ejecuta:

```sh
python -m venv .venv
```

Activa el entorno con `.venv\Scripts\activate.bat` en Windows (CMD) o `source .venv/bin/activate` en macOS y Linux. Luego ejecuta:

```sh
python -m pip install -r requirements.txt
python app.py
```

Abre la dirección local que muestra Streamlit en la terminal. En Windows también puedes hacer doble clic en `ABRIR_QUANTLAB.cmd` para preparar el entorno y abrir la interfaz.

## Qué incluye esta edición

- La interfaz V3 con sus diez vistas y la documentación de la reformulación.
- Hasta 767 velas recientes por contrato e intervalo para Mercado, más las velas de 15 minutos que utiliza el detalle de las operaciones de margen.
- Indicadores para la ventana de Mercado y los registros de escenarios, además de las columnas que utilizan las vistas de episodios y resultados.
- Los informes seleccionados y los resultados congelados de ambas carteras. Los resultados del ensayo se conservan al reducir los archivos de entrada.

Esta edición muestra una instantánea histórica y sirve para abrir el panel directamente. El conjunto completo de datos, los checkpoints y el entorno de entrenamiento quedan en la carpeta original del proyecto. `snapshot_manifest.json` enumera los archivos de código y datos de la instantánea.
