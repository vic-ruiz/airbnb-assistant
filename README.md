
# 🌟 Airbnb Assistant – RAG + LLM + Ollama
Asistente inteligente para responder consultas de huéspedes basado en **RAG**, **Ollama** y **Streamlit**.

---

## ✨ Descripción general

Esta herramienta permite pegar el mensaje de un huésped y obtener automáticamente:

- Recuperación de información según la propiedad (RAG)
- Identificación de intención (check-in, disponibilidad, amenities, reglas, etc.)
- Extracción y normalización de fechas
- Verificación de disponibilidad real vía **iCal**
- Redacción automática de un mensaje amable y listo para enviar

💬 Ideal para anfitriones que manejan múltiples propiedades y quieren agilizar el flujo de respuestas.

---

## 📁 Estructura del proyecto

```
airbnb-assistant/
│
├── app.py                 # UI + orquestación Streamlit
├── generator.py           # prompts + cliente Ollama
├── retriever.py           # motor RAG (FAISS + SQLite)
├── kb_build.py            # construye la KB (faiss.index + kb.sqlite)
├── ical_utils.py          # funciones para leer .ics y validar disponibilidad
├── check_ical_demo.py     # script opcional para probar iCal
│
├── data/
│   ├── kb.jsonl           # Base de conocimiento editable ✔
│   ├── faiss.index        # Índice FAISS (GENERADO) ❌ no subir al repo
│   ├── kb.sqlite          # Base SQLite (GENERADA) ❌ no subir al repo
│
├── .env                   # Variables privadas ❌ no subir
├── .env.example           # Plantilla ✔
├── requirements.txt       # Dependencias
├── .gitignore             # Exclusiones sensibles
└── README.md              # Documentación
```

---

## 🧩 Requisitos

### ✔ Python 3.11  
Verificar versión:

```bash
python --version
```

### ✔ Instalar Ollama  
Descarga:  
https://ollama.com/download

### ✔ Modelo recomendado

```
qwen2.5:3b-instruct
```

Instalar modelo:

```bash
ollama pull qwen2.5:3b-instruct
```

Iniciar Ollama:

```bash
ollama serve
```

---

# 🚀 Instalación y ejecución

## 1) Clonar el repositorio

```bash
git clone https://github.com/noencrp87/airbnb-assistant.git
cd airbnb-assistant
```

---

## 2) Crear entorno virtual (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

---

## 3) Instalar dependencias

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 4) Crear archivo `.env`

Duplicar la plantilla:

```bash
cp .env.example .env
```

Editar `.env` con:

```
# Ollama
OLLAMA_HOST="http://localhost:11434"
OLLAMA_MODEL="qwen2.5:3b-instruct"

# iCal (URLs exportadas desde Airbnb)
ICAL_RECOLETA="URL_ICS_RECOLETA"
ICAL_PARAGUAY="URL_ICS_PARAGUAY"
```

---

## 5) Construir la Base de Conocimiento (KB)

Este comando **debe ejecutarse cada vez que edites `kb.jsonl`**:

```bash
python kb_build.py
```

Esto genera:

- `data/faiss.index`
- `data/kb.sqlite`

---

## 6) Ejecutar la aplicación

```bash
python -m streamlit run app.py
```

La app se abrirá en:  
👉 **http://localhost:8501**

---

# 📚 Cómo editar la Base de Conocimiento (RAG)

El archivo principal es:

```
data/kb.jsonl
```

Formato válido: **un JSON por línea**.

Ejemplo:

```json
{"property_id": "MICRO-PARAGUAY-870", "section": "checkin", "lang": "es", "text": "Check-in a partir de las 15:00."}
```

Luego correr:

```bash
python kb_build.py
```

---

# 🧪 Probar funcionalidad iCal

Ver eventos del calendario y validar disponibilidad:

```bash
python check_ical_demo.py
```

---

# 🔒 Buenas prácticas / Seguridad

El repositorio **NO debe incluir**:

- `.env`
- `.venv/`
- `data/faiss.index`
- `data/kb.sqlite`
- `__pycache__/`

Todo esto está gestionado en `.gitignore`.

---

# ❗ Errores comunes y soluciones

### 🔴 “model not found”
No bajaste el modelo:

```bash
ollama pull qwen2.5:3b-instruct
```

---

### 🔴 “Could not open data/faiss.index”
Te falta correr:

```bash
python kb_build.py
```

---

### 🔴 “ModuleNotFoundError” (faiss, dateparser, sentence-transformers, etc.)
Ejecutar:

```bash
pip install -r requirements.txt
```

---

### 🔴 Streamlit no abre
Cerrar consola → abrir nueva → activar venv → ejecutar:

```bash
python -m streamlit run app.py
```

---

# 👩‍💻 Créditos

Proyecto desarrollado por  
**Jablonski - Ramírez - Ruiz – Maestría en Ciencia de Datos – Universidad Austral**  
