"""
app.py
======
Clasificador de Dependencias MEN — Aplicación web (Streamlit)
ESEIT: Escuela Superior de Empresa, Ingeniería y Tecnología
Autores: Gabriela Contreras Cañas | Andrés Felipe Zárate Chaparro

DESCRIPCIÓN:
  Interfaz web que permite a cualquier persona, sin instalar nada,
  escribir el Asunto de una comunicación ciudadana y recibir las
  3 dependencias más probables del MEN con su porcentaje de confianza.
"""

import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────
RUTA_MODELO = Path(__file__).parent / "modelo_final.pkl"

MIN_PALABRAS = 2
ETIQUETA_REVISION = "Asunto insuficiente – revisión manual"

GENERICOS = {
    'solicitud', 'petición', 'peticion', 'consulta', 'información', 'informacion',
    'queja', 'reclamo', 'derecho de petición', 'derecho de peticion', 'tutela',
    'pqrs', 'pqr', 'comunicación', 'comunicacion', 'oficio', 'carta',
    'petición sin asunto', 'peticion sin asunto', 'sin asunto', 'n/a', 'na',
    '.', '-', 'ninguno', '', 'urgente', 'importante', 'fwd:', 're:',
    'solicitud urgente', 'solicitud información', 'solicitud informacion',
}

ACTIVAR_COHERENCIA = True
FACTOR_PENALIZACION = 0.35

KW_SUPERIOR = [
    'universidad', 'universitario', 'universitaria', 'educación superior',
    'pregrado', 'posgrado', 'postgrado', 'maestría', 'magíster', 'doctorado',
    'ies', 'institución de educación superior', 'instituciones de educación superior',
    'convalidación', 'convalidar', 'título profesional', 'rector', 'rectoría',
    'acreditación', 'snies', 'icetex', 'educación técnica', 'técnico profesional',
    'tecnológico', 'sena', 'programa académico', 'créditos académicos',
]

KW_BASICA = [
    'colegio', 'escuela', 'preescolar', 'primaria', 'bachillerato', 'bachiller',
    'básica', 'secundaria', 'media vocacional', 'niño', 'niña', 'niños', 'niñas',
    'primera infancia', 'jardín', 'jardines', 'docente de aula', 'profesor de colegio',
    'institución educativa', 'establecimiento educativo', 'pae', 'alimentación escolar',
    'matrícula escolar', 'grado escolar', 'menor de edad', 'menores de edad',
    'preescolares', 'transición', 'estudiante de grado',
]

VERTICE_DEP = {
    'Despacho del Ministro': 'Ministro',
    'Oficina Asesora de Comunicaciones': 'Ministro',
    'Oficina Asesora Jurídica': 'Ministro',
    'Oficina Asesora de Planeación y Finanzas': 'Ministro',
    'Oficina de Control Interno': 'Ministro',
    'Oficina de Tecnología y Sistemas de Información': 'Ministro',
    'Oficina de Cooperación y Asuntos Internacionales': 'Ministro',
    'Oficina de Innovación Educativa con Uso de Nuevas Tecnologías': 'Ministro',
    'Oficina de Control Disciplinario Interno': 'Ministro',
    'Oficina de Infraestructura Educativa': 'Ministro',
    'Dirección de Calidad para la Educación Preescolar Básica y Media': 'VicePBM',
    'Despacho del Viceministro(a) de Educación Preescolar Básica y Media': 'VicePBM',
    'Subdirección de Referentes y Evaluación Educativa': 'VicePBM',
    'Subdirección de Fomento de Competencias': 'VicePBM',
    'Dirección de Fortalecimiento a la Gestión Territorial': 'VicePBM',
    'Subdirección de Monitoreo y Control': 'VicePBM',
    'Subdirección de Recursos Humanos del Sector educación': 'VicePBM',
    'Subdirección de Fortalecimiento Institucional': 'VicePBM',
    'Dirección de Cobertura y Equidad': 'VicePBM',
    'Subdirección de Permanencia': 'VicePBM',
    'Subdirección de Acceso': 'VicePBM',
    'Dirección de Primera Infancia': 'VicePBM',
    'Subdirección de Cobertura de Primera Infancia': 'VicePBM',
    'Subdirección de Calidad de Primera Infancia': 'VicePBM',
    'Despacho del Viceministro(a) de Educación Superior': 'ViceES',
    'Dirección de Calidad para la Educación Superior': 'ViceES',
    'Subdirección de Inspección y Vigilancia': 'ViceES',
    'Subdirección de Aseguramiento de la Educación Superior': 'ViceES',
    'Dirección de Fomento de la Educación Superior': 'ViceES',
    'Subdirección de Apoyo a la Gestión de las Instituciones de Educación Superior': 'ViceES',
    'Subdirección de Desarrollo Sectorial': 'ViceES',
    'Secretaría General': 'SecGen',
    'Subdirección de Relacionamiento con la Ciudadanía': 'SecGen',
    'Subdirección de Gestión Financiera': 'SecGen',
    'Subdirección de Desarrollo Organizacional': 'SecGen',
    'Subdirección de Talento Humano': 'SecGen',
    'Subdirección de Contratación': 'SecGen',
    'Subdirección de Gestión Administrativa': 'SecGen',
}

VERTICE_NOMBRES = {
    'Ministro': 'Despacho del Ministro',
    'VicePBM':  'Vice Educación Preescolar, Básica y Media',
    'ViceES':   'Vice Educación Superior',
    'SecGen':   'Secretaría General',
}

VERTICE_COLOR = {
    'Ministro': '#534AB7',
    'VicePBM':  '#378ADD',
    'ViceES':   '#D85A30',
    'SecGen':   '#1D9E75',
}


# ─────────────────────────────────────────────────────────────
# FUNCIONES
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    with open(RUTA_MODELO, 'rb') as f:
        return pickle.load(f)


def limpiar(texto: str) -> str:
    if not isinstance(texto, str):
        return ''
    t = re.sub(r'\d{4}-[A-Z]+-\d+', '', texto)
    t = re.sub(r'\s+', ' ', t)
    return t.strip()


def es_ruidoso(texto: str) -> bool:
    t = limpiar(texto)
    if len(t.split()) < MIN_PALABRAS:
        return True
    if t.lower() in GENERICOS:
        return True
    return False


def detectar_nivel(texto: str):
    t = texto.lower()
    n_sup = sum(1 for k in KW_SUPERIOR if k in t)
    n_bas = sum(1 for k in KW_BASICA if k in t)
    if n_sup > n_bas and n_sup > 0:
        return 'ViceES'
    if n_bas > n_sup and n_bas > 0:
        return 'VicePBM'
    return None


def top3_dependencias(asunto: str, modelo: dict) -> list:
    if es_ruidoso(asunto):
        return []

    vec = modelo['vec']
    clf = modelo['modelo']

    texto = limpiar(asunto)
    X_vec = vec.transform([texto])
    probas = clf.predict_proba(X_vec)[0].copy()

    if ACTIVAR_COHERENCIA:
        nivel = detectar_nivel(texto)
        if nivel is not None:
            opuesto = 'VicePBM' if nivel == 'ViceES' else 'ViceES'
            for i, cls in enumerate(clf.classes_):
                if VERTICE_DEP.get(str(cls).strip()) == opuesto:
                    probas[i] *= FACTOR_PENALIZACION
            total = probas.sum()
            if total > 0:
                probas = probas / total

    top_idx = np.argsort(probas)[::-1][:3]

    resultados = []
    for idx in top_idx:
        dep = str(clf.classes_[idx]).strip()
        pct = round(float(probas[idx]) * 100, 1)
        if pct > 0:
            resultados.append((dep, pct))

    return resultados


# ─────────────────────────────────────────────────────────────
# INTERFAZ
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clasificador de Dependencias MEN",
    page_icon="🏛️",
    layout="centered",
)

st.title("Clasificador de Dependencias")
st.caption(
    "ESEIT: Escuela Superior de Empresa, Ingeniería y Tecnología · "
    "Proyecto Aplicado II  \n"
    "Implementación de un modelo de Procesamiento de Lenguaje Natural (NLP) "
    "para la clasificación de peticiones ciudadanas basado en el Decreto 2269 de 2023  \n"
)

st.write(
    "Escribe el **asunto** de una comunicación ciudadana y el modelo "
    "indicará a qué dependencia del MEN podría corresponder."
)

try:
    modelo = cargar_modelo()
except FileNotFoundError:
    st.error(
        "No se encontró el archivo `modelo_final.pkl` en el repositorio. "
        "Verifica que se haya subido junto con `app.py`."
    )
    st.stop()

tab1, tab2 = st.tabs(["Consulta individual", "Procesamiento por lotes (Excel)"])


# ── TAB 1: consulta individual ──────────────────────────────────────────────
with tab1:
    ejemplos = [
        "Solicitud de convalidación de título universitario obtenido en el exterior",
        "Solicitud de revisión a institución de educación superior por acoso",
        "Reclamo sobre alimentación escolar PAE en colegio público",
        "Permiso remunerado por motivos personales para empleado del MEN",
        "Notificación auto admite acción de tutela contra el MEN",
    ]

    st.write("**Ejemplos rápidos:**")
    cols = st.columns(len(ejemplos))
    for i, ej in enumerate(ejemplos):
        with cols[i]:
            if st.button(f"Ejemplo {i+1}", help=ej, use_container_width=True):
                st.session_state["asunto_texto"] = ej

    asunto = st.text_area(
        "Asunto de la comunicación",
        key="asunto_texto",
        height=100,
        placeholder="Ej: Solicitud de convalidación de título universitario obtenido en el exterior",
    )

    if st.button("Clasificar", type="primary"):
        if not asunto.strip():
            st.warning("Por favor escribe el asunto de la comunicación.")
        else:
            candidatos = top3_dependencias(asunto, modelo)

            if not candidatos:
                st.warning(
                    f"**{ETIQUETA_REVISION}**\n\n"
                    "El asunto no contiene suficiente información para clasificarlo "
                    "automáticamente. Se recomienda revisión manual."
                )
            else:
                st.markdown("#### La solicitud puede ser direccionada a:")
                for i, (dep, pct) in enumerate(candidatos, 1):
                    vert = VERTICE_DEP.get(dep, 'Ministro')
                    color = VERTICE_COLOR.get(vert, '#888780')
                    nombre_vert = VERTICE_NOMBRES.get(vert, '')

                    c1, c2 = st.columns([5, 1])
                    with c1:
                        st.markdown(f"**{i}. {dep}**")
                        st.caption(nombre_vert)
                    with c2:
                        st.markdown(
                            f"<div style='text-align:right; font-size:22px; "
                            f"font-weight:600; color:{color};'>{pct}%</div>",
                            unsafe_allow_html=True,
                        )
                    st.progress(min(pct / 100, 1.0))

                top1_pct = candidatos[0][1]
                if top1_pct < 30:
                    st.info(
                        "Confianza baja en la primera opción — se recomienda "
                        "revisión manual antes de enrutar."
                    )


# ── TAB 2: procesamiento por lotes ──────────────────────────────────────────
with tab2:
    st.write(
        "Sube un archivo Excel con una columna llamada **Asunto** "
        "(los datos pueden empezar en la fila 2 o más adelante)."
    )

    # Plantilla descargable
    from io import BytesIO as _BytesIO
    _plantilla = pd.DataFrame({
        "Asunto": [
            "Solicitud de convalidación de título universitario obtenido en el exterior",
            "Reclamo sobre alimentación escolar PAE en colegio público",
            "Permiso remunerado por motivos personales para empleado del MEN",
        ]
    })
    _buf_plantilla = _BytesIO()
    with pd.ExcelWriter(_buf_plantilla, engine='openpyxl') as _writer:
        _plantilla.to_excel(_writer, index=False, sheet_name="Asuntos")
    _buf_plantilla.seek(0)

    st.download_button(
        label="Descargar plantilla de ejemplo (Excel)",
        data=_buf_plantilla,
        file_name="plantilla_asuntos.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    archivo = st.file_uploader("Archivo Excel (.xlsx)", type=["xlsx"])

    columna = st.text_input("Nombre de la columna con el asunto", value="Asunto")

    if archivo is not None:
        try:
            df_in = pd.read_excel(archivo)
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}")
            st.stop()

        if columna not in df_in.columns:
            st.error(
                f"La columna '{columna}' no existe en el archivo. "
                f"Columnas disponibles: {', '.join(df_in.columns.astype(str))}"
            )
        else:
            if st.button("Procesar archivo", type="primary"):
                asuntos = df_in[columna].fillna('').tolist()
                progreso = st.progress(0, text="Clasificando...")

                op1, p1, op2, p2, op3, p3, estado = [], [], [], [], [], [], []

                for idx, asunto in enumerate(asuntos):
                    candidatos = top3_dependencias(asunto, modelo)
                    if not candidatos:
                        op1.append(ETIQUETA_REVISION); p1.append(0.0)
                        op2.append(''); p2.append(np.nan)
                        op3.append(''); p3.append(np.nan)
                        estado.append('Revisión manual')
                    else:
                        c = candidatos + [('', np.nan)] * (3 - len(candidatos))
                        op1.append(c[0][0]); p1.append(c[0][1])
                        op2.append(c[1][0]); p2.append(c[1][1])
                        op3.append(c[2][0]); p3.append(c[2][1])

                        top1_pct = candidatos[0][1]
                        if top1_pct >= 50:
                            estado.append('Confianza Alta')
                        elif top1_pct >= 30:
                            estado.append('Confianza Media')
                        else:
                            estado.append('Confianza Baja')

                    if len(asuntos) > 0:
                        progreso.progress((idx + 1) / len(asuntos))

                progreso.empty()

                df_out = df_in.copy()
                df_out['Dependencia_01']    = op1
                df_out['Porc_01']           = p1
                df_out['Dependencia_02']    = op2
                df_out['Porc_02']           = p2
                df_out['Dependencia_03']    = op3
                df_out['Porc_03']           = p3
                df_out['Estado']            = estado

                st.success(f"Procesamiento completo: {len(df_out):,} registros.")

                resumen = pd.Series(estado).value_counts()
                st.write("**Resumen:**")
                st.dataframe(resumen.rename("Cantidad"), use_container_width=True)

                st.dataframe(df_out, use_container_width=True)

                # Descargar resultado
                from io import BytesIO
                buffer = BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    df_out.to_excel(writer, index=False)
                buffer.seek(0)

                st.download_button(
                    label="Descargar resultados (Excel)",
                    data=buffer,
                    file_name="Comunicaciones_clasificadas.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )


st.divider()
st.caption(
    "Modelo: TF-IDF + SGDClassifier, entrenado con ~403.000 comunicaciones "
    "(sep 2025 – jun 2026) filtradas por Etapa = Aceptar. \n "
    "De las 37 dependencias normalizadas, 12 tienen F1 ≥ 0.60 (buen desempeño), "
    "10 están entre 0.40-0.60 (mejorable), y 15 quedan por debajo de 0.40 "
    "(críticas, generalmente por tener pocas muestras de entrenamiento). \n"
    "El Accuracy de 85.3% es la cifra más representativa para presentar de "
    "forma general, ya que refleja el desempeño global ponderado por volumen "
    "real de comunicaciones. \n"
)
st.caption(
    "Esta herramienta corresponde a un ejercicio académico desarrollado en el "
    "marco de un proyecto de especialización y se basa en la estructura "
    "orgánica establecida en el Decreto 2269 de 2023. No constituye un acto "
    "administrativo, ni reemplaza ni representa el procedimiento oficial de "
    "recepción, trámite y asignación de comunicaciones ciudadanas vigente "
    "en el MEN.  \n"
)
