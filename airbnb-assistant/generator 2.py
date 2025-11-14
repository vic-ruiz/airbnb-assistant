# generator.py - VERSIÓN CON VALIDACIÓN ESTRICTA DE FACTS
from __future__ import annotations

import json
import re
import requests
from typing import List, Dict, Any, Optional

OLLAMA_API = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:3b-instruct"
DEFAULT_TEMPERATURE = 0.1  # Más bajo para mayor consistencia

def render_ctx_snippets(snippets: List[Dict[str, Any]]) -> str:
    """Formatea snippets del RAG."""
    if not snippets:
        return "(sin fragmentos de contexto)"
    
    lines = []
    for i, s in enumerate(snippets, start=1):
        pid = s.get("property_id", "N/A")
        sec = s.get("section", "N/A")
        txt = s.get("text", "").strip().replace("\n", " ")
        
        if len(txt) > 150:
            txt = txt[:147] + "..."
        
        lines.append(f"[{i}] {pid} | {sec}: {txt}")
    
    return "\n".join(lines[:8])

# =========================
# PROMPT ULTRA ESTRICTO
# =========================

SYSTEM_PROMPT = """Eres un asistente para anfitriones de Airbnb. Respondes consultas de huéspedes de forma profesional, cálida y PRECISA.

⚠️ REGLA CRÍTICA #1: RESPONDE SOLO LO QUE TE PREGUNTAN
- Si preguntan por WIFI, habla de WIFI (no de disponibilidad)
- Si preguntan por GYM, habla de GYM (no de disponibilidad)  
- Si preguntan por PRECIO, habla de PRECIO (no de disponibilidad)
- Solo menciona disponibilidad si hay HECHOS VERIFICADOS sobre disponibilidad

⚠️ REGLA CRÍTICA #2: PRIORIDAD ABSOLUTA DE HECHOS VERIFICADOS
Los HECHOS VERIFICADOS son información validada en tiempo real (ej: iCal).
- Si un HECHO dice "✅ DISPONIBLE" → Confirma disponibilidad
- Si un HECHO dice "❌ NO DISPONIBLE" → Comunica que NO está disponible
- Si NO hay hechos de disponibilidad → NO hables de disponibilidad

⚠️ REGLA CRÍTICA #3: RESPUESTAS ENFOCADAS
- Para amenities: Responde sobre las comodidades específicas que preguntaron
- Para pricing: Explica que necesitas fechas para cotizar
- Para checkin/checkout: Da información de horarios
- NO mezcles temas si no los preguntaron

CONTEXTO RAG:
- Úsalo como referencia (políticas, amenities, ubicación)
- Cita específicamente lo que sirve para la pregunta

TONO:
- Profesional pero cercano
- Español rioplatense (Argentina)
- Conciso, sin redundancia

FORMATO DE SALIDA:
Devuelve SOLO este JSON:
{
  "intent": "amenities",
  "dates": [],
  "draft": "Respuesta directa a la pregunta",
  "citations": ["fuente1"],
  "language": "es"
}
"""

USER_TEMPLATE = """=== MENSAJE DEL HUÉSPED ===
{email_text}

=== PROPIEDAD ===
{property_id}

=== 🔴 HECHOS VERIFICADOS (MÁXIMA PRIORIDAD) ===
{facts_text}

=== CONTEXTO DE REFERENCIA (solo si no contradice hechos) ===
{ctx_text}

=== PARÁMETROS ===
Tono: {style}
Firma: {signature}

=== INSTRUCCIONES ===
1. Lee los HECHOS VERIFICADOS primero
2. Si hay información de disponibilidad ahí, úsala EXACTAMENTE como está
3. Genera un "draft" que sea coherente con los hechos
4. Devuelve SOLO el JSON (sin ```json ni texto adicional)
"""

def _facts_to_text(extra_facts: Optional[List[str]]) -> str:
    """Formatea hechos verificados con máximo énfasis."""
    if not extra_facts:
        return "❌ NO HAY HECHOS VERIFICADOS\n(Usa solo el contexto de referencia)"
    
    # Destacar hechos de disponibilidad con formato ultra visible
    formatted = []
    for fact in extra_facts[:6]:
        if "DISPONIBLE" in fact.upper():
            if "✅" in fact or "DISPONIBLE del" in fact:
                formatted.append(f"🟢🟢🟢 {fact} 🟢🟢🟢")
                formatted.append("→ CONFIRMA DISPONIBILIDAD EN TU RESPUESTA")
            elif "❌" in fact or "NO DISPONIBLE" in fact:
                formatted.append(f"🔴🔴🔴 {fact} 🔴🔴🔴")
                formatted.append("→ COMUNICA QUE NO ESTÁ DISPONIBLE")
        else:
            formatted.append(f"ℹ️ {fact}")
    
    return "\n".join(formatted)

def _call_ollama(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Llama a Ollama con configuración optimizada."""
    body = {
        "model": model,
        "format": "json",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "temperature": temperature,
            "num_predict": 800,
            "top_p": 0.9,
            **({"seed": seed} if seed is not None else {}),
        },
        "stream": False,
    }
    
    try:
        r = requests.post(OLLAMA_API, json=body, timeout=120)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        raise RuntimeError(f"Error conectando con Ollama: {e}")
    
    content = data.get("message", {}).get("content", "").strip()
    
    if not content:
        raise RuntimeError("Ollama no devolvió contenido")
    
    # Parsear JSON con limpieza agresiva
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Intentar limpiar markdown
        cleaned = content.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            # Último recurso
            return {
                "intent": "other",
                "dates": [],
                "draft": "Disculpá, tuve un problema técnico. ¿Podrías reformular tu consulta?",
                "citations": [],
                "language": "es",
                "_raw_error": content[:200],
                "_parse_error": str(e)
            }

def _validate_availability_consistency(
    draft: str,
    extra_facts: Optional[List[str]]
) -> Tuple[bool, Optional[str]]:
    """
    Valida que el draft sea consistente con los hechos de disponibilidad.
    Retorna: (es_válido, draft_corregido_o_None)
    """
    if not extra_facts:
        return True, None
    
    # Buscar hechos de disponibilidad
    has_available_fact = False
    has_not_available_fact = False
    
    for fact in extra_facts:
        fact_upper = fact.upper()
        if "✅" in fact or ("DISPONIBLE" in fact_upper and "NO DISPONIBLE" not in fact_upper):
            has_available_fact = True
        elif "❌" in fact or "NO DISPONIBLE" in fact_upper:
            has_not_available_fact = True
    
    # Si no hay hechos de disponibilidad, no validar
    if not has_available_fact and not has_not_available_fact:
        return True, None
    
    draft_lower = draft.lower()
    
    # Palabras que indican "no disponible"
    negative_indicators = [
        "no está disponible", "no disponible", "no hay disponibilidad",
        "ocupadas", "ocupada", "reservadas", "reservada",
        "lamento", "lamentablemente", "desafortunadamente"
    ]
    
    # Palabras que indican "disponible"
    positive_indicators = [
        "disponible", "libre", "confirmo", "confirmamos",
        "está libre", "hay lugar", "pueden reservar"
    ]
    
    draft_says_available = any(pos in draft_lower for pos in positive_indicators)
    draft_says_not_available = any(neg in draft_lower for neg in negative_indicators)
    
    # VALIDACIÓN ESTRICTA
    if has_available_fact:
        # El hecho dice DISPONIBLE
        if draft_says_not_available and not draft_says_available:
            # ¡CONTRADICCIÓN! El draft dice NO disponible pero el hecho dice SÍ
            # Generar draft corregido
            fact_text = [f for f in extra_facts if "DISPONIBLE" in f.upper()][0]
            corrected = f"""¡Buenas noticias! {fact_text.replace('[VERIFICADO_ICAL]', '').replace('✅', '').strip()}

Para confirmar la reserva, necesitaría:
- Cantidad de huéspedes
- Confirmar las fechas exactas

¿Te parece si avanzamos con la reserva?

Saludos,
Equipo de Atención"""
            return False, corrected
    
    elif has_not_available_fact:
        # El hecho dice NO DISPONIBLE
        if draft_says_available and not draft_says_not_available:
            # ¡CONTRADICCIÓN! El draft dice disponible pero el hecho dice NO
            fact_text = [f for f in extra_facts if "NO DISPONIBLE" in f.upper()][0]
            corrected = f"""Lamento informarte que {fact_text.replace('[VERIFICADO_ICAL]', '').replace('❌', '').strip()}

¿Te gustaría que te sugiera fechas alternativas cercanas? También puedo anotarte en una lista de espera por si hay cancelación.

Saludos,
Equipo de Atención"""
            return False, corrected
    
    return True, None

def generate_with_llm(
    *,
    email_text: str,
    property_id: Optional[str],
    ctx_snippets: List[Dict[str, Any]],
    style: str = "cálido",
    signature: str = "Equipo de Atención",
    seed: Optional[int] = 7,
    extra_facts: Optional[List[str]] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Dict[str, Any]:
    """
    Genera respuesta con LLM + validación estricta de consistencia.
    """
    ctx_text = render_ctx_snippets(ctx_snippets)
    facts_text = _facts_to_text(extra_facts)
    
    user_prompt = USER_TEMPLATE.format(
        email_text=email_text.strip(),
        property_id=property_id or "(sin especificar)",
        ctx_text=ctx_text,
        facts_text=facts_text,
        style=style,
        signature=signature,
    )
    
    # Primera llamada al LLM
    raw_output = _call_ollama(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        temperature=temperature,
        seed=seed,
    )
    
    # Extraer campos
    intent = (raw_output.get("intent") or "other").strip().lower()
    dates = raw_output.get("dates") or []
    draft = raw_output.get("draft") or ""
    citations = raw_output.get("citations") or []
    language = (raw_output.get("language") or "es").strip().lower()
    
    # VALIDACIÓN CRÍTICA: Verificar consistencia con hechos
    if intent == "availability" and draft:
        is_valid, corrected_draft = _validate_availability_consistency(draft, extra_facts)
        
        if not is_valid and corrected_draft:
            # ¡Contradicción detectada! Usar draft corregido
            draft = corrected_draft
            citations.insert(0, "⚠️ Respuesta corregida por inconsistencia con hechos verificados")
    
    # Validaciones de tipos
    if not isinstance(dates, list):
        dates = []
    if not isinstance(citations, list):
        citations = []
    
    # Filtrar fechas inválidas
    valid_dates = []
    for d in dates:
        if isinstance(d, str) and len(d) == 10 and d.count("-") == 2:
            valid_dates.append(d)
    
    # Si el draft está vacío, generar uno básico
    if not draft.strip():
        draft = f"Gracias por tu consulta. Estamos revisando tu mensaje y te respondemos a la brevedad.\n\nSaludos,\n{signature}"
    
    return {
        "intent": intent,
        "dates": valid_dates,
        "draft": draft,
        "citations": citations[:6],
        "language": language,
        "_debug": {
            "raw_output": raw_output,
            "facts_provided": len(extra_facts) if extra_facts else 0,
            "validation_passed": is_valid if intent == "availability" else None
        }
    }

def validate_llm_response(response: Dict[str, Any]) -> tuple[bool, List[str]]:
    """Valida estructura de respuesta del LLM."""
    errors = []
    required_fields = ["intent", "dates", "draft", "citations", "language"]
    
    for field in required_fields:
        if field not in response:
            errors.append(f"Campo faltante: {field}")
    
    if "draft" in response and len(response["draft"]) < 10:
        errors.append("Draft demasiado corto")
    
    return len(errors) == 0, errors
