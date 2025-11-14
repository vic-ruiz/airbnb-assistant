# -*- coding: utf-8 -*-
# generator_multi_intent.py - VERSION MULTI-INTENCION
from __future__ import annotations

import json
import re
import requests
from typing import List, Dict, Any, Optional, Set, Tuple

OLLAMA_API = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:3b-instruct"
DEFAULT_TEMPERATURE = 0.1

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
# PROMPT MULTI-INTENCION
# =========================

SYSTEM_PROMPT = """Eres un asistente para anfitriones de Airbnb. Respondes consultas de huespedes de forma profesional, calida y PRECISA.

REGLA FUNDAMENTAL: RESPONDE **TODAS** LAS PREGUNTAS DEL HUESPED
Si el huesped pregunta varias cosas (ej: "esta disponible del 2/2 al 5/2? hay wifi?"), debes responder AMBAS preguntas en el mismo mensaje.

REGLA CRITICA 1: PRIORIDAD ABSOLUTA DE HECHOS VERIFICADOS
Los HECHOS VERIFICADOS son informacion validada en tiempo real (ej: iCal).
- Si un HECHO dice "DISPONIBLE del X al Y" -> Confirma disponibilidad EXACTAMENTE con esas fechas
- Si un HECHO dice "NO DISPONIBLE" -> Comunica que NO esta disponible
- Si NO hay hechos de disponibilidad -> NO inventes informacion de disponibilidad

REGLA CRITICA 2: ESTRUCTURA DE RESPUESTA MULTI-PREGUNTA
Cuando el huesped pregunta varias cosas:
1. Responde cada pregunta en parrafos separados
2. Se claro y directo con cada respuesta
3. NO omitas ninguna pregunta

Ejemplo:
Pregunta: "esta disponible del 2/2 al 5/2? hay wifi?"
Respuesta correcta:
"Respecto a la disponibilidad, te confirmo que [INFO DE DISPONIBILIDAD VERIFICADA].

En cuanto al WiFi, si, el departamento cuenta con conexion WiFi de alta velocidad."

REGLA CRITICA 3: NUNCA INVENTES DATOS
- Si NO tienes informacion verificada de disponibilidad, di "necesito verificar disponibilidad"
- Si NO tienes informacion de amenities, di "voy a consultar eso y te respondo"
- NUNCA digas "esta disponible" sin un HECHO VERIFICADO que lo confirme

CONTEXTO RAG:
- Usalo como referencia (politicas, amenities, ubicacion)
- Cita especificamente lo que sirve para cada pregunta

TONO:
- Profesional pero cercano
- Espanol rioplatense (Argentina)
- Conciso pero completo

FORMATO DE SALIDA:
Devuelve SOLO este JSON:
{
  "intent": "availability,amenities",
  "dates": ["2026-02-02"],
  "draft": "Respuesta que aborda TODAS las preguntas",
  "citations": ["fuente1", "fuente2"],
  "language": "es"
}
"""

USER_TEMPLATE = """=== MENSAJE DEL HUESPED ===
{email_text}

=== PROPIEDAD ===
{property_id}

=== HECHOS VERIFICADOS (MAXIMA PRIORIDAD) ===
{facts_text}

=== CONTEXTO DE REFERENCIA (solo si no contradice hechos) ===
{ctx_text}

=== PARAMETROS ===
Tono: {style}
Firma: {signature}

=== IMPORTANTE ===
El sistema detecto estas intenciones en el mensaje: {intents}
Asegurarte de responder TODAS las preguntas del huesped relacionadas con estas intenciones.

=== INSTRUCCIONES ===
1. Lee los HECHOS VERIFICADOS primero
2. Si hay informacion de disponibilidad ahi, usala EXACTAMENTE como esta
3. Identifica TODAS las preguntas del huesped
4. Genera un "draft" que responda CADA pregunta claramente
5. Devuelve SOLO el JSON (sin ```json ni texto adicional)
"""

def _facts_to_text(extra_facts: Optional[List[str]]) -> str:
    """Formatea hechos verificados con maximo enfasis."""
    if not extra_facts:
        return "NO HAY HECHOS VERIFICADOS\n(Usa solo el contexto de referencia)"
    
    formatted = []
    for fact in extra_facts[:6]:
        if "DISPONIBLE" in fact.upper():
            if "DISPONIBLE del" in fact and "NO DISPONIBLE" not in fact:
                formatted.append(f">>> {fact} <<<")
                formatted.append("-> CONFIRMA DISPONIBILIDAD EN TU RESPUESTA")
            elif "NO DISPONIBLE" in fact:
                formatted.append(f">>> {fact} <<<")
                formatted.append("-> COMUNICA QUE NO ESTA DISPONIBLE")
        else:
            formatted.append(f"INFO: {fact}")
    
    return "\n".join(formatted)

def _call_ollama(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Llama a Ollama con configuracion optimizada."""
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
        raise RuntimeError("Ollama no devolvio contenido")
    
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        cleaned = content.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            return {
                "intent": "other",
                "dates": [],
                "draft": "Disculpa, tuve un problema tecnico. Podrias reformular tu consulta?",
                "citations": [],
                "language": "es",
                "_raw_error": content[:200],
                "_parse_error": str(e)
            }

def _validate_multi_intent_response(
    draft: str,
    extra_facts: Optional[List[str]],
    intents: Optional[Set[str]]
) -> Tuple[bool, Optional[str]]:
    """
    Valida que el draft responda TODAS las intenciones detectadas.
    Y que sea consistente con los hechos verificados.
    """
    if not draft or not intents:
        return True, None
    
    draft_lower = draft.lower()
    # Normalizar acentos
    draft_lower = draft_lower.replace('á', 'a').replace('é', 'e').replace('í', 'i')
    draft_lower = draft_lower.replace('ó', 'o').replace('ú', 'u')
    
    issues = []
    
    # 1) Validar disponibilidad si hay hechos verificados
    if extra_facts and ("availability" in intents):
        has_available_fact = any("DISPONIBLE del" in f and "NO DISPONIBLE" not in f for f in extra_facts)
        has_not_available_fact = any("NO DISPONIBLE" in f for f in extra_facts)
        
        negative_indicators = ["no esta disponible", "no disponible", "no hay disponibilidad", "ocupadas", "ocupada", "reservadas", "reservada", "lamento"]
        positive_indicators = ["disponible", "libre", "confirmo", "confirmamos", "esta libre", "hay lugar"]
        
        draft_says_available = any(pos in draft_lower for pos in positive_indicators)
        draft_says_not_available = any(neg in draft_lower for neg in negative_indicators)
        
        if has_available_fact and draft_says_not_available and not draft_says_available:
            issues.append("CONTRADICCION: Dice NO disponible pero el hecho dice SI")
        
        if has_not_available_fact and draft_says_available and not draft_says_not_available:
            issues.append("CONTRADICCION: Dice disponible pero el hecho dice NO")
    
    # 2) Validar que responda a amenities si esta en intents
    if "amenities" in intents:
        amenities_words = ["wifi", "gym", "gimnasio", "pileta", "piscina", "toallas", "cocina", "calefacci", "aire"]
        has_amenities_mention = any(word in draft_lower for word in amenities_words)
        
        if not has_amenities_mention:
            issues.append("NO RESPONDIO: Pregunta sobre amenities sin respuesta")
    
    # 3) Validar que responda a pricing si esta en intents
    if "pricing" in intents:
        pricing_words = ["precio", "tarifa", "costo", "necesit", "fechas", "cotiz"]
        has_pricing_mention = any(word in draft_lower for word in pricing_words)
        
        if not has_pricing_mention:
            issues.append("NO RESPONDIO: Pregunta sobre precio sin respuesta")
    
    if not issues:
        return True, None
    
    # Generar respuesta corregida si hay problemas
    corrected_parts = []
    
    if "availability" in intents and extra_facts:
        avail_fact = [f for f in extra_facts if "DISPONIBLE" in f.upper()]
        if avail_fact:
            fact_clean = avail_fact[0].replace('[VERIFICADO_ICAL]', '').strip()
            corrected_parts.append(f"Respecto a la disponibilidad: {fact_clean}")
    
    if "amenities" in intents:
        corrected_parts.append("En cuanto a las comodidades del lugar, necesito verificar esa informacion especifica. Te respondo en breve.")
    
    if "pricing" in intents:
        corrected_parts.append("Para darte un precio exacto necesito confirmar fechas y cantidad de huespedes. Me pasas esos datos?")
    
    corrected = "\n\n".join(corrected_parts) + f"\n\nSaludos,\nEquipo de Atencion"
    
    return False, corrected

def generate_with_llm(
    *,
    email_text: str,
    property_id: Optional[str],
    ctx_snippets: List[Dict[str, Any]],
    style: str = "calido",
    signature: str = "Equipo de Atencion",
    seed: Optional[int] = 7,
    extra_facts: Optional[List[str]] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    intents: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Genera respuesta con LLM + validacion estricta de consistencia.
    Soporta multiples intenciones simultaneas.
    """
    ctx_text = render_ctx_snippets(ctx_snippets)
    facts_text = _facts_to_text(extra_facts)
    
    # Formatear intenciones para el prompt
    intents_str = ", ".join(sorted(intents)) if intents else "no detectadas"
    
    user_prompt = USER_TEMPLATE.format(
        email_text=email_text.strip(),
        property_id=property_id or "(sin especificar)",
        ctx_text=ctx_text,
        facts_text=facts_text,
        style=style,
        signature=signature,
        intents=intents_str,
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
    
    # VALIDACION CRITICA: Verificar consistencia con hechos y cobertura de intenciones
    is_valid, corrected_draft = _validate_multi_intent_response(draft, extra_facts, intents)
    
    if not is_valid and corrected_draft:
        # Problemas detectados! Usar draft corregido
        draft = corrected_draft
        citations.insert(0, "Respuesta corregida por inconsistencias detectadas")
    
    # Validaciones de tipos
    if not isinstance(dates, list):
        dates = []
    if not isinstance(citations, list):
        citations = []
    
    # Filtrar fechas invalidas
    valid_dates = []
    for d in dates:
        if isinstance(d, str) and len(d) == 10 and d.count("-") == 2:
            valid_dates.append(d)
    
    # Si el draft esta vacio, generar uno basico
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
            "intents_requested": list(intents) if intents else [],
            "validation_passed": is_valid
        }
    }

def validate_llm_response(response: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Valida estructura de respuesta del LLM."""
    errors = []
    required_fields = ["intent", "dates", "draft", "citations", "language"]
    
    for field in required_fields:
        if field not in response:
            errors.append(f"Campo faltante: {field}")
    
    if "draft" in response and len(response["draft"]) < 10:
        errors.append("Draft demasiado corto")
    
    return len(errors) == 0, errors
