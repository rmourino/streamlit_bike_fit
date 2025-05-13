
import streamlit as st
from PIL import Image
import mediapipe as mp
import numpy as np
import pandas as pd
from fpdf import FPDF
from io import BytesIO
import tempfile

# Configuración de la página
st.set_page_config(
    page_title="Bike Fitting con análisis de postura",
    page_icon="📐",
    layout="wide",
)

# # Encabezado visual y título
# st.image(
#     "https://upload.wikimedia.org/wikipedia/commons/2/28/Cycling_posture_diagram.jpg",
#     use_column_width=True,
#     caption="Postura ciclista ideal"
# )
st.markdown("<h1 style='text-align: center; color: #2E86AB;'>🚴 Bike Fitting con análisis de postura</h1>", unsafe_allow_html=True)

# Formulario compacto
col1, col2 = st.columns(2)
with col1:
    lado = st.radio("Lado del cuerpo:", ("Derecho", "Izquierdo"), horizontal=True)
with col2:
    opcion_img = st.radio("Método:", ("📸 Cámara", "📂 Subir imagen"), horizontal=True)

img_data = None
if opcion_img == "📸 Cámara":
    img_data = st.camera_input("Toma una foto de tu postura")
elif opcion_img == "📂 Subir imagen":
    img_data = st.file_uploader("Sube una imagen (JPG, PNG)", type=["jpg", "jpeg", "png"])

if img_data:
    image = Image.open(img_data).convert("RGB")
    image_np = np.array(image)

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_np)

        if results.pose_landmarks:
            image_draw = image.copy()
            image_draw_np = np.array(image_draw)
            mp_drawing.draw_landmarks(
                image=image_draw_np,
                landmark_list=results.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            st.image(Image.fromarray(image_draw_np), caption="Postura detectada", use_column_width=True)

            lm = results.pose_landmarks.landmark
            side = "RIGHT" if lado == "Derecho" else "LEFT"
            def p(j): return [lm[getattr(mp_pose.PoseLandmark, f"{side}_{j}").value].x,
                              lm[getattr(mp_pose.PoseLandmark, f"{side}_{j}").value].y]

            def get_angle(a, b, c):
                a, b, c = np.array(a), np.array(b), np.array(c)
                angle = np.arccos(np.clip(np.dot(b - a, c - b) /
                                          (np.linalg.norm(b - a) * np.linalg.norm(c - b)), -1.0, 1.0))
                return int(np.degrees(angle))

            angulos = {}
            recomendaciones = []

            try:
                angulos["Codo"] = get_angle(p("SHOULDER"), p("ELBOW"), p("WRIST"))
                angulos["Hombro"] = get_angle(p("ELBOW"), p("SHOULDER"), p("HIP"))
                angulos["Cadera"] = get_angle(p("SHOULDER"), p("HIP"), p("KNEE"))
                angulos["Rodilla"] = get_angle(p("HIP"), p("KNEE"), p("ANKLE"))
                angulos["Tobillo"] = get_angle(p("KNEE"), p("ANKLE"), p("FOOT_INDEX"))
            except Exception as e:
                st.error(f"Error calculando ángulos: {e}")

            datos_export = []
            def clean_text(text):
                return (
                    text.replace("→", "->")
                        .replace("°", " grados")
                        .replace("á", "a")
                        .replace("é", "e")
                        .replace("í", "i")
                        .replace("ó", "o")
                        .replace("ú", "u")
                        .replace("ñ", "n")
                )

            for k, v in angulos.items():
                msg = f"{k}: {v}° → "
                if k == "Rodilla":
                    if v < 70:
                        rec = "muy cerrado -> subir el asiento."
                    elif v > 150:
                        rec = "muy abierto -> bajar el asiento."
                    else:
                        rec = "adecuado."
                elif k == "Codo":
                    if v < 100:
                        rec = "muy flexionado -> ajustar manillar."
                    elif v > 160:
                        rec = "demasiado extendido."
                    else:
                        rec = "correcto."
                elif k == "Hombro":
                    rec = "tension posible." if v < 40 else "razonable."
                elif k == "Cadera":
                    rec = "cerrado -> revisar sillon/manillar." if v < 70 else "correcto."
                elif k == "Tobillo":
                    rec = "fuera de rango -> revisar tecnica." if v < 80 or v > 120 else "correcto."
                msg += rec
                recomendaciones.append(clean_text(msg))
                datos_export.append({"Ángulo": k, "Valor": v, "Recomendación": rec})
                st.markdown(f"**{k}**: {v}° — _{rec}_")

            df = pd.DataFrame(datos_export)

            st.download_button(
                label="📄 Descargar CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="bikefitting_resultados.csv",
                mime="text/csv"
            )

            # Guardar imagen temporal
            annotated_img = Image.fromarray(image_draw_np)
            temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            annotated_img.save(temp_img.name)
            final_img_path = temp_img.name

            # PDF
            class PDF(FPDF):
                def header(self):
                    self.set_font("Arial", "B", 14)
                    self.cell(0, 10, "Informe de Bike Fitting", ln=True, align="C")
                    self.ln(10)
                def body(self, recomendaciones, img_path):
                    self.set_font("Arial", "", 12)
                    for line in recomendaciones:
                        self.multi_cell(0, 10, line)
                        self.ln(1)
                    if img_path:
                        self.ln(5)
                        self.image(img_path, x=30, w=150)

            pdf = PDF()
            pdf.add_page()
            pdf.body(recomendaciones, final_img_path)
            pdf_output = BytesIO()
            pdf_bytes = pdf.output(dest="S").encode("latin-1", errors="ignore")
            pdf_output.write(pdf_bytes)
            pdf_output.seek(0)

            st.download_button(
                label="📄 Descargar PDF",
                data=pdf_output,
                file_name="bikefitting_informe.pdf",
                mime="application/pdf"
            )
        else:
            st.error("No se detectó postura. Intenta con mejor iluminación y perfil lateral claro.")
