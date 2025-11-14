# -*- coding: utf-8 -*-
# app_multi_intent.py - VERSION CON MULTIPLES INTENCIONES
import re
import sqlite3
from datetime import date, timedelta
from typing import List, Tuple, Optional, Set

import streamlit as st
from langdetect import detect
from jinja2 import Template
from dotenv import load_dotenv
import os

from retriever import Retriever
from generator import generate_with_llm

load_dotenv()

st.set_page_config(page_title="Asistente Airbnb - RAG + LLM", layout="wide")

# =========================
# PARSER DE FECHAS
# =========================

class UltraRobustDateParser:
    """Parser que maneja TODO tipo de expresion de fechas en español."""
    
    WEEKDAY_NAMES = {
        'lunes': 0, 'martes': 1, 'miercoles': 2,
        'jueves': 3, 'viernes': 4, 'sabado': 5, 'domingo': 6
    }
    
    MONTH_NAMES = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
        'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
        'septiembre': 9, 'setiembre': 9, 'octubre': 10,
        'noviembre': 11, 'diciembre': 12,
        'ene': 1, 'feb': 2, 'mar': 3, 'abr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'ago': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dic': 12
    }
    
    @staticmethod
    def normalize(text: str) -> str:
        t = text.lower()
        # Normalizar acentos
        t = t.replace('á', 'a').replace('é', 'e').replace('í', 'i')
        t = t.replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
        t = re.sub(r'\s+', ' ', t).strip()
        return t
    
    @staticmethod
    def infer_year_and_month(day: int, month: Optional[int], today: date) -> Tuple[int, int]:
        if month is None:
            month = today.month
        
        year = today.year
        
        if month < today.month:
            year += 1
        elif month == today.month and day < today.day:
            year += 1
        
        return year, month
    
    @classmethod
    def parse_numeric_dates(cls, text: str, today: date) -> List[Tuple[date, date]]:
        t = cls.normalize(text)
        ranges = []
        
        pattern = r'\b(\d{1,2})[/\-](\d{1,2})\b'
        matches = re.finditer(pattern, t)
        
        found_dates = []
        for match in matches:
            day, month = int(match.group(1)), int(match.group(2))
            
            if 1 <= day <= 31 and 1 <= month <= 12:
                year, month = cls.infer_year_and_month(day, month, today)
                try:
                    found_dates.append(date(year, month, day))
                except ValueError:
                    continue
        
        if len(found_dates) == 1:
            ranges.append((found_dates[0], found_dates[0] + timedelta(days=1)))
        elif len(found_dates) >= 2:
            found_dates.sort()
            ranges.append((found_dates[0], found_dates[-1]))
        
        return ranges
    
    @classmethod
    def parse_explicit_dates(cls, text: str, today: date) -> List[Tuple[date, date]]:
        t = cls.normalize(text)
        ranges = []
        
        pattern_desde = r'\b(?:desde|a\s*partir\s*de)(?:\s+el)?\s+(\d{1,2})(?:ro|do|er|vo|to|st|nd|rd|th)?\s+(?:de\s+)?(\w+)\b'
        match = re.search(pattern_desde, t)
        if match:
            day = int(match.group(1))
            month_name = match.group(2)
            
            if month_name in cls.MONTH_NAMES:
                month = cls.MONTH_NAMES[month_name]
                year, month = cls.infer_year_and_month(day, month, today)
                
                try:
                    checkin = date(year, month, day)
                    nights = 1
                    night_match = re.search(r'(\d+)\s*noche(s)?', t)
                    if night_match:
                        nights = int(night_match.group(1))
                    checkout = checkin + timedelta(days=nights)
                    ranges.append((checkin, checkout))
                    return ranges
                except ValueError:
                    pass
        
        pattern_el = r'\b(?:el|disponible\s+el?)\s+(\d{1,2})(?:ro|do|er|vo|to)?\s+(?:de\s+)?(\w+)\b'
        match = re.search(pattern_el, t)
        if match:
            day = int(match.group(1))
            month_name = match.group(2)
            
            if month_name in cls.MONTH_NAMES:
                month = cls.MONTH_NAMES[month_name]
                year, month = cls.infer_year_and_month(day, month, today)
                
                try:
                    checkin = date(year, month, day)
                    checkout = checkin + timedelta(days=1)
                    ranges.append((checkin, checkout))
                    return ranges
                except ValueError:
                    pass
        
        pattern_del_al = r'\b(?:del|desde)\s+(\d{1,2})\s+(?:al|hasta)\s+(?:el\s+)?(\d{1,2})\s*(?:de\s+)?(\w+)?\b'
        match = re.search(pattern_del_al, t)
        if match:
            day1, day2 = int(match.group(1)), int(match.group(2))
            month_name = match.group(3)
            
            month = cls.MONTH_NAMES.get(month_name) if month_name else today.month
            year, month = cls.infer_year_and_month(day1, month, today)
            
            try:
                checkin = date(year, month, day1)
                checkout = date(year, month, day2)
                if checkout > checkin:
                    ranges.append((checkin, checkout))
                    return ranges
            except ValueError:
                pass
        
        pattern_ir_del = r'\bir\s+del\s+(\d{1,2})\s+(?:de\s+)?(\w+)\s+al\s+(\d{1,2})\b'
        match = re.search(pattern_ir_del, t)
        if match:
            day1 = int(match.group(1))
            month_name = match.group(2)
            day2 = int(match.group(3))
            
            if month_name in cls.MONTH_NAMES:
                month = cls.MONTH_NAMES[month_name]
                year, month = cls.infer_year_and_month(day1, month, today)
                
                try:
                    checkin = date(year, month, day1)
                    checkout = date(year, month, day2)
                    if checkout > checkin:
                        ranges.append((checkin, checkout))
                        return ranges
                except ValueError:
                    pass
        
        pattern_semana = r'\b(?:primer|primera)\s+semana\s+(?:de\s+)?(\w+)\b'
        match = re.search(pattern_semana, t)
        if match:
            month_name = match.group(1)
            if month_name in cls.MONTH_NAMES:
                month = cls.MONTH_NAMES[month_name]
                year = today.year if month >= today.month else today.year + 1
                
                try:
                    checkin = date(year, month, 1)
                    checkout = date(year, month, 7)
                    ranges.append((checkin, checkout))
                    return ranges
                except ValueError:
                    pass
        
        return ranges
    
    @classmethod
    def parse_relative_dates(cls, text: str, today: date) -> List[Tuple[date, date]]:
        t = cls.normalize(text)
        ranges = []
        
        if re.search(r'\b(?:el\s+)?fin(?:de)?(?:\s+de\s+semana)?(?:\s+que\s+viene|\s+proximo)?\b', t):
            days_ahead = (4 - today.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            checkin = today + timedelta(days=days_ahead)
            checkout = checkin + timedelta(days=2)
            ranges.append((checkin, checkout))
            return ranges
        
        if re.search(r'\b(?:la\s+)?semana\s+que\s+viene\b', t):
            days_to_monday = (7 - today.weekday()) % 7
            if days_to_monday == 0:
                days_to_monday = 7
            checkin = today + timedelta(days=days_to_monday)
            checkout = checkin + timedelta(days=2)
            ranges.append((checkin, checkout))
            return ranges
        
        for day_name, weekday in cls.WEEKDAY_NAMES.items():
            if re.search(rf'\b{day_name}\s+que\s+viene\b', t):
                days_ahead = (weekday - today.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7
                checkin = today + timedelta(days=days_ahead)
                checkout = checkin + timedelta(days=1)
                ranges.append((checkin, checkout))
                return ranges
        
        return ranges
    
    @classmethod
    def parse_all(cls, text: str, today: Optional[date] = None) -> List[Tuple[date, date]]:
        if today is None:
            today = date.today()
        
        ranges = cls.parse_explicit_dates(text, today)
        if ranges:
            return ranges
        
        ranges = cls.parse_numeric_dates(text, today)
        if ranges:
            return ranges
        
        ranges = cls.parse_relative_dates(text, today)
        if ranges:
            return ranges
        
        return []

# =========================
# CLASIFICACION MULTI-INTENCION
# =========================

def classify_intents_multi(text: str, has_dates: bool) -> Set[str]:
    """
    Detecta TODAS las intenciones en el mensaje.
    Retorna un Set con todas las intenciones encontradas.
    """
    t = UltraRobustDateParser.normalize(text)
    intents = set()
    
    # 1. AMENITIES
    amenities_patterns = [
        r'\bgym\b', r'\bgimnasio\b', r'\bwifi\b', r'\bwi\s*fi\b',
        r'\btoalla(s)?\b', r'\bsabana(s)?\b', r'\bcocina\b',
        r'\bpileta\b', r'\bpiscina\b', r'\bair(e)?\s*acondicionado\b',
        r'\bcalefaccion\b', r'\bestacionamiento\b', r'\bgarage\b',
        r'\bamenities\b', r'\bservicios\b', r'\bequipamiento\b'
    ]
    
    for pattern in amenities_patterns:
        if re.search(pattern, t):
            intents.add("amenities")
            break
    
    # 2. AVAILABILITY (si hay fechas)
    if has_dates:
        availability_patterns = [
            r'\bdisponibl(e|idad)\b', r'\breserv(ar|a|as)?\b',
            r'\bhay\s+lugar\b', r'\blibre\b', r'\bpuedo\s+(?:reservar|ir)\b',
            r'\besta\s+(?:disponible|libre)\b'
        ]
        
        for pattern in availability_patterns:
            if re.search(pattern, t):
                intents.add("availability")
                break
        
        # Si hay fechas y no se detecto ninguna otra intencion, es availability implicito
        if not intents:
            intents.add("availability")
    
    # 3. PRICING (solo si NO hay fechas, para evitar conflicto)
    if not has_dates:
        pricing_patterns = [
            r'\bprecio(s)?\b', r'\btarifa(s)?\b', r'\bcosto(s)?\b',
            r'\bcuanto\s+(?:cuesta|sale|es)\b', r'\bvalor\b'
        ]
        
        for pattern in pricing_patterns:
            if re.search(pattern, t):
                intents.add("pricing")
                break
    
    # 4. CHECK-IN/OUT
    if re.search(r'\bcheck\s*in\b|\bingreso\b|\bllegada\b', t):
        intents.add("checkin")
    
    if re.search(r'\bcheck\s*out\b|\bsalida\b|\begreso\b', t):
        intents.add("checkout")
    
    # 5. RECOMMENDATIONS
    if re.search(r'\brecomendacion(es)?\b|\bdonde\s+comer\b|\bque\s+hacer\b', t):
        intents.add("recommendations")
    
    # 6. POLICY
    if re.search(r'\bcancelacion\b|\bnorma(s)?\b|\bpolitica(s)?\b', t):
        intents.add("policy")
    
    # Si no se detecto ninguna intencion, marcar como "other"
    if not intents:
        intents.add("other")
    
    return intents

# =========================
# VALIDADOR DE DISPONIBILIDAD
# =========================

def check_availability_robust(
    email_text: str,
    property_id: Optional[str],
    ical_url: str,
    today: Optional[date] = None
) -> dict:
    from ical_utils import is_available
    
    if today is None:
        today = date.today()
    
    result = {
        "success": False,
        "message": "",
        "checkin": None,
        "checkout": None,
        "available": None
    }
    
    if not property_id:
        result["message"] = "Necesito saber a que propiedad te referis."
        return result
    
    if not ical_url:
        result["message"] = f"No puedo verificar disponibilidad de {property_id}."
        return result
    
    parser = UltraRobustDateParser()
    ranges = parser.parse_all(email_text, today)
    
    if not ranges:
        result["message"] = "Necesito fechas de check-in y check-out."
        return result
    
    checkin, checkout = ranges[0]
    result["checkin"] = checkin
    result["checkout"] = checkout
    
    if checkout <= checkin:
        result["message"] = "Check-out debe ser posterior a check-in."
        return result
    
    if checkin < today:
        result["message"] = f"La fecha {checkin.strftime('%d/%m/%Y')} ya paso."
        return result
    
    try:
        ical_result = is_available(ical_url, checkin, checkout)
        result["success"] = True
        result["available"] = ical_result["available"]
        
        nights = (checkout - checkin).days
        
        if ical_result["available"]:
            result["message"] = (
                f"DISPONIBLE del {checkin.strftime('%d/%m/%Y')} al {checkout.strftime('%d/%m/%Y')} "
                f"({nights} noche{'s' if nights > 1 else ''})."
            )
        else:
            conflicts = ical_result.get("conflicts", [])
            if conflicts:
                c = conflicts[0]
                result["message"] = (
                    f"NO DISPONIBLE. Hay una reserva del {c['start'][:10]} al {c['end'][:10]}."
                )
            else:
                result["message"] = "NO DISPONIBLE en esas fechas."
    
    except Exception as e:
        result["message"] = f"Error: {str(e)}"
    
    return result

# =========================
# UTILIDADES
# =========================

def detect_lang(text: str):
    try:
        return detect(text)
    except Exception:
        return "es"

@st.cache_data
def load_property_ids(db_path="data/kb.sqlite"):
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT property_id FROM kb ORDER BY property_id;")
        props = [r[0] for r in cur.fetchall()]
        conn.close()
        return props
    except Exception:
        return []

@st.cache_resource
def get_retriever():
    return Retriever()

def get_ical_url(property_id: str) -> str:
    if not property_id:
        return ""
    mapping = {
        "RECOLETA-PATIO": os.environ.get("ICAL_RECOLETA", ""),
        "MICRO-PARAGUAY-870": os.environ.get("ICAL_PARAGUAY", ""),
    }
    return mapping.get(property_id, "")

# =========================
# UI
# =========================
st.title("Asistente para anfitriones – RAG + LLM local (Ollama)")
st.caption("MVP de Text Mining/NLP: IR (FAISS), clasificación de intención, generación con grounding.")

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    email_text = st.text_area(
        "Correo del huesped:",
        height=260,
        placeholder="Ej: esta disponible del 2/2 al 5/2? hay wifi?"
    )
    run = st.button("Procesar", use_container_width=True)

with col2:
    st.subheader("Parametros")
    signature = st.text_input("Firma", value="Equipo de Atencion")
    props = load_property_ids()
    options = ["(sin filtro)"] + props if props else ["(sin filtro)"]
    property_id_choice = st.selectbox("Propiedad", options=options, index=1 if len(options) > 1 else 0)
    property_id = None if property_id_choice == "(sin filtro)" else property_id_choice
    use_llm = st.checkbox("Usar LLM", value=True)

if run and email_text.strip():
    today = date.today()
    retr = get_retriever()
    
    # 1) PARSEO DE FECHAS
    parser = UltraRobustDateParser()
    date_ranges = parser.parse_all(email_text, today)
    has_dates = len(date_ranges) > 0
    
    st.markdown("### Analisis")
    if date_ranges:
        for checkin, checkout in date_ranges:
            nights = (checkout - checkin).days
            st.success(f"**Fechas:** {checkin.strftime('%d/%m/%Y')} -> {checkout.strftime('%d/%m/%Y')} ({nights} noche{'s' if nights > 1 else ''})")
    else:
        st.info("Sin fechas detectadas")
    
    # 2) CLASIFICACION MULTI-INTENCION
    intents = classify_intents_multi(email_text, has_dates)
    lang = detect_lang(email_text)
    
    st.write(f"- **Intenciones:** `{', '.join(sorted(intents))}`")
    st.write(f"- **Idioma:** {lang}")
    
    # 3) CONTEXTO RAG
    ctx_chunks = retr.retrieve(email_text, k=6, property_id=property_id)
    
    # 4) VERIFICACION DE DISPONIBILIDAD (SI HAY FECHAS)
    availability_result = None
    if has_dates:
        ical_url = get_ical_url(property_id)
        availability_result = check_availability_robust(email_text, property_id, ical_url, today)
        
        st.markdown("### Disponibilidad Verificada")
        if availability_result["success"]:
            if availability_result["available"]:
                st.success(availability_result["message"])
            else:
                st.error(availability_result["message"])
        else:
            st.warning(availability_result["message"])
    
    # 5) GENERACION CON LLM
    draft = ""
    citations = []
    validation_warning = None
    
    if use_llm:
        try:
            extra_facts = []
            
            # Agregar disponibilidad verificada como HECHO
            if availability_result and availability_result.get("success"):
                extra_facts.append(f"[VERIFICADO_ICAL] {availability_result['message']}")
            
            # Informar al LLM sobre las intenciones detectadas
            intents_str = ", ".join(sorted(intents))
            extra_facts.append(f"[INTENCIONES_DETECTADAS] {intents_str}")
            
            llm_response = generate_with_llm(
                email_text=email_text,
                property_id=property_id,
                ctx_snippets=ctx_chunks,
                style="calido",
                signature=signature,
                seed=7,
                extra_facts=extra_facts,
                intents=intents
            )
            
            draft = llm_response.get("draft", "")
            citations = llm_response.get("citations", [])
            
            debug_info = llm_response.get("_debug", {})
            if debug_info.get("validation_passed") == False:
                validation_warning = "Respuesta corregida por inconsistencia."
            
        except Exception as e:
            st.warning(f"LLM fallo: {e}")
            draft = f"Disculpa, hubo un error tecnico. Podrias reformular tu consulta?\n\nSaludos,\n{signature}"
    else:
        # Fallback simple si no usa LLM
        ctx_summary = " ".join([f"{c['text'][:100]}..." for c in ctx_chunks[:2]])
        draft = f"Gracias por tu consulta.\n\n{ctx_summary}\n\nSaludos,\n{signature}"
    
    # 6) OUTPUT
    st.markdown("### Respuesta Generada")
    
    if validation_warning:
        st.warning(validation_warning)
    
    st.text_area("Borrador", draft, height=300, key="output")
    
    # Mostrar contexto RAG
    if ctx_chunks:
        with st.expander("Contexto RAG"):
            for i, ch in enumerate(ctx_chunks, start=1):
                st.markdown(f"**[{i}]** {ch.get('section', 'N/A')}")
                st.write(ch.get("text", "")[:150])
    
    # Debug de intenciones
    with st.expander("Debug Multi-Intent"):
        st.json({
            "intents_detected": list(intents),
            "has_dates": has_dates,
            "availability_checked": availability_result is not None,
            "availability_result": availability_result["message"] if availability_result else None
        })

# Debug parser
with st.expander("Debug Parser"):
    test_text = st.text_input("Texto:", "esta disponible del 2/2 al 5/2? hay wifi?")
    if st.button("Analizar"):
        parser = UltraRobustDateParser()
        test_ranges = parser.parse_all(test_text)
        test_intents = classify_intents_multi(test_text, len(test_ranges) > 0)
        
        st.write("**Fechas:**")
        if test_ranges:
            for ci, co in test_ranges:
                st.success(f"OK {ci.strftime('%d/%m/%Y')} -> {co.strftime('%d/%m/%Y')}")
        else:
            st.error("Sin fechas")
        
        st.write("**Intenciones:**")
        for intent in sorted(test_intents):
            st.info(intent)
