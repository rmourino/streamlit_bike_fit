import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
import time
import pandas as pd
from fpdf import FPDF
from io import BytesIO
from PIL import Image
import tempfile

st.set_page_config(
    page_title="Bike Fitting con análisis de postura",
    page_icon="📐",
    layout="wide"
)
# st.title("📐 Bike Fitting con análisis de postura")

lado = st.radio("¿Qué lado estás mostrando a la cámara?", ("Derecho", "Izquierdo"))

if st.button("Iniciar análisis"):
    st.write("Preparándote...")
    countdown = st.empty()
    for i in range(3, 0, -1):
        countdown.write(f"{i}...")
        time.sleep(1)
    countdown.write("¡Grabando!")

    duration = 5
    cap = cv2.VideoCapture(0)
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    start_time = time.time()
    all_angles = {k: [] for k in ["codo", "hombro", "cadera", "rodilla", "tobillo"]}
    last_frame = None
    last_landmarks = None
    image_placeholder = st.empty()

    def get_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        angle = np.arccos(np.clip(np.dot(b - a, c - b) /
                                  (np.linalg.norm(b - a) * np.linalg.norm(c - b)), -1.0, 1.0))
        return int(np.degrees(angle))

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            st.error("Error con la cámara.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            side = "RIGHT" if lado == "Derecho" else "LEFT"
            def p(j): return [lm[getattr(mp_pose.PoseLandmark, f"{side}_{j}").value].x,
                                lm[getattr(mp_pose.PoseLandmark, f"{side}_{j}").value].y]

            try:
                all_angles["codo"].append(get_angle(p("SHOULDER"), p("ELBOW"), p("WRIST")))
                all_angles["hombro"].append(get_angle(p("ELBOW"), p("SHOULDER"), p("HIP")))
                all_angles["cadera"].append(get_angle(p("SHOULDER"), p("HIP"), p("KNEE")))
                all_angles["rodilla"].append(get_angle(p("HIP"), p("KNEE"), p("ANKLE")))
                all_angles["tobillo"].append(get_angle(p("KNEE"), p("ANKLE"), p("FOOT_INDEX")))
                last_frame = frame.copy()
                last_landmarks = results.pose_landmarks
            except:
                pass

            # Mostrar frame en tiempo real con puntos
            frame_display = frame.copy()
            mp_drawing.draw_landmarks(
                image=frame_display,
                landmark_list=results.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            frame_display = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
            image_placeholder.image(frame_display, channels="RGB", use_column_width=True)

    countdown.empty()
    image_placeholder.empty()
    
    final_img_path = None
    if last_frame is not None and last_landmarks:
        annotated = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
        mp_drawing.draw_landmarks(
            image=annotated,
            landmark_list=last_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )
        st.image(annotated, caption="Postura capturada", use_column_width=True)

        # Guardar imagen temporal
        img = Image.fromarray(annotated)
        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        img.save(temp_img.name)
        final_img_path = temp_img.name

    cap.release()
    pose.close()

    st.success("Grabación finalizada")

    # Mostrar imagen final
    if last_frame is not None and last_landmarks:
        annotated = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
        mp_drawing.draw_landmarks(
            image=annotated,
            landmark_list=last_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )
        st.image(annotated, caption="Postura capturada", use_column_width=True)

    # Mostrar análisis
    resumen = []
    datos_export = []

    for k, values in all_angles.items():
        if values:
            avg = int(np.mean(values))
            minimo = min(values)
            maximo = max(values)

            st.markdown(f"### Ángulo de {k.capitalize()}")
            st.write(f"Mínimo: {minimo}° — Máximo: {maximo}° — Promedio: {avg}°")

            msg = f"El ángulo promedio de {k} es {avg}°. "
            if k == "rodilla":
                if avg < 70:
                    msg += "El ángulo es demasiado cerrado. Sugerimos subir el asiento."
                elif avg > 150:
                    msg += "El ángulo es muy abierto. Sugerimos bajar el asiento."
                else:
                    msg += "El ángulo parece adecuado."
            elif k == "codo":
                if avg < 100:
                    msg += "Muy flexionado. Revisa la posición del manillar."
                elif avg > 160:
                    msg += "Demasiado extendido. Puede causar fatiga."
                else:
                    msg += "Ángulo funcional."
            elif k == "hombro":
                if avg < 40:
                    msg += "Ángulo cerrado. Puede generar tensión en cuello/hombros."
                else:
                    msg += "Ángulo razonable."
            elif k == "cadera":
                if avg < 70:
                    msg += "Muy cerrado. Revisa altura del manillar o retroceso del sillín."
                else:
                    msg += "Buena flexión de cadera."
            elif k == "tobillo":
                if avg < 80 or avg > 120:
                    msg += "Movimiento fuera del rango. Revisa técnica o calas."
                else:
                    msg += "Movimiento dentro del rango típico."

            st.info(msg)
            resumen.append(msg)
            datos_export.append({
                "Ángulo": k.capitalize(),
                "Mínimo": minimo,
                "Máximo": maximo,
                "Promedio": avg,
                "Recomendación": msg
            })
        else:
            st.warning(f"No se pudo calcular el ángulo de {k}")

    # Exportar CSV
    df = pd.DataFrame(datos_export)
    st.download_button(
        label="📄 Descargar resultados en CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="resultados_bikefitting.csv",
        mime="text/csv"
    )

    # Exportar PDF
    class PDF(FPDF):
        def header(self):
            self.set_font("Arial", "B", 14)
            self.cell(0, 10, "Informe de Bike Fitting", ln=True, align="C")
        def body(self, resumen):
            self.set_font("Arial", "", 12)
            self.ln(10)
            for line in resumen:
                self.multi_cell(0, 10, line)
                self.ln(1)
        def add_image(self, img_path):
            self.ln(5)
            self.image(img_path, x=30, w=150)

    pdf = PDF()
    pdf.add_page()
    pdf.body(resumen)
    if final_img_path:
        pdf.add_image(final_img_path)
    pdf_output = BytesIO()
    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    pdf_output.write(pdf_bytes)
    pdf_output.seek(0)

    st.download_button(
        label="📄 Descargar informe en PDF",
        data=pdf_output,
        file_name="informe_bikefitting.pdf",
        mime="application/pdf"
    )
