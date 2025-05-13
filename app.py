import streamlit as st
from PIL import Image
import mediapipe as mp
import numpy as np
import pandas as pd
from fpdf import FPDF
from io import BytesIO
import cv2
import tempfile
import math

# Configuración de página
st.set_page_config(
    page_title="Bike Fitting con análisis de postura",
    page_icon="📐",
    layout="wide",
)

st.markdown("<h1 style='text-align: center; color: #2E86AB;'>🚴 Bike Fitting con análisis de postura</h1>", unsafe_allow_html=True)

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
    image = Image.open(img_data)
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image=image_np,
                landmark_list=results.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )

            st.image(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), caption="Postura detectada", use_column_width=True)

            lm = results.pose_landmarks.landmark
            side = "RIGHT" if lado == "Derecho" else "LEFT"
            def p(j): return [lm[getattr(mp_pose.PoseLandmark, f"{side}_{j}").value].x,
                                    lm[getattr(mp_pose.PoseLandmark, f"{side}_{j}").value].y]

            # def get_angle(a, b, c):
            #     a, b, c = np.array(a), np.array(b), np.array(c)
            #     angle = np.arccos(np.clip(np.dot(b - a, c - b) /
            #                               (np.linalg.norm(b - a) * np.linalg.norm(c - b)), -1.0, 1.0))
            #     return int(np.degrees(angle))
            
            def get_angle(a, b, c):
                ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
                return ang + 360 if ang < 0 else ang

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
            for k, v in angulos.items():
                msg = f"{k}: {v}° — "
                if k == "Rodilla":
                    if v < 70:
                        rec = "muy cerrado → subir el asiento."
                    elif v > 150:
                        rec = "muy abierto → bajar el asiento."
                    else:
                        rec = "adecuado."
                elif k == "Codo":
                    if v < 100:
                        rec = "muy flexionado → ajustar manillar."
                    elif v > 160:
                        rec = "demasiado extendido."
                    else:
                        rec = "correcto."
                elif k == "Hombro":
                    rec = "tensión posible." if v < 40 else "razonable."
                elif k == "Cadera":
                    rec = "cerrado → revisar sillín/manillar." if v < 70 else "correcto."
                elif k == "Tobillo":
                    rec = "fuera de rango → revisar técnica." if v < 80 or v > 120 else "correcto."
                recomendaciones.append(f"{k}: {v}° → {rec}")
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
            img = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
            temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            img.save(temp_img.name)
            final_img_path = temp_img.name

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

            # PDF
            class PDF(FPDF):
                def header(self):
                    self.set_font("Arial", "B", 14)
                    self.cell(0, 10, "Informe de Bike Fitting", ln=True, align="C")
                    self.ln(10)

                def body(self, recomendaciones, img_path):
                    self.set_font("Arial", "", 12)
                    for line in recomendaciones:
                        self.multi_cell(0, 10, clean_text(line))
                        self.ln(1)
                    if img_path:
                        self.ln(5)
                        self.image(img_path, x=30, w=150)

            pdf = PDF()
            pdf.add_page()
            pdf.body(recomendaciones, final_img_path)
            pdf_output = BytesIO()
            pdf_bytes = pdf.output(dest="S").encode("latin-1")
            pdf_output.write(pdf_bytes)
            pdf_output.seek(0)

            st.download_button(
                label="📄 Descargar PDF",
                data=pdf_output,
                file_name="bikefitting_informe.pdf",
                mime="application/pdf"
            )

        else:
            st.error("No se detectó ninguna postura. Intenta con mejor iluminación y muestra el perfil.")
