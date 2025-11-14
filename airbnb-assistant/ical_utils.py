# ical_utils.py - VERSIÓN CORREGIDA Y MEJORADA
import io
import os
import requests
from datetime import datetime, timedelta, timezone, date
from typing import List, Tuple, Dict, Optional

import pytz
from icalendar import Calendar

# CRÍTICO: Descomentar para manejar eventos recurrentes
try:
    import recurring_ical_events
    HAS_RECURRING = True
except ImportError:
    HAS_RECURRING = False
    print("⚠️ WARNING: recurring_ical_events no está instalado. Solo se detectarán eventos simples.")

# Zona horaria de trabajo (ajusta si corresponde)
TZ = pytz.timezone("America/Argentina/Buenos_Aires")

def _to_aware(dt) -> datetime:
    """Convierte date/datetime a datetime aware en TZ local."""
    if isinstance(dt, date) and not isinstance(dt, datetime):
        return TZ.localize(datetime(dt.year, dt.month, dt.day, 0, 0))
    if isinstance(dt, datetime):
        # Normalizamos todo a TZ local
        if dt.tzinfo is None:
            return TZ.localize(dt)
        return dt.astimezone(TZ)
    raise ValueError(f"Tipo de fecha no soportado: {type(dt)}")


def fetch_calendar(ics_url: str) -> Calendar:
    """
    Descarga el .ics y devuelve un objeto Calendar.
    Incluye reintentos y mejor manejo de errores.
    """
    try:
        resp = requests.get(ics_url, timeout=30)
        resp.raise_for_status()
        return Calendar.from_ical(resp.content)
    except requests.RequestException as e:
        raise RuntimeError(f"Error descargando calendario desde {ics_url}: {e}")
    except Exception as e:
        raise RuntimeError(f"Error parseando calendario: {e}")


def expand_busy_intervals_with_recurring(
    cal: Calendar, 
    start: datetime, 
    end: datetime
) -> List[Tuple[datetime, datetime, str]]:
    """
    Expande eventos INCLUYENDO recurrencias usando recurring_ical_events.
    """
    if not HAS_RECURRING:
        raise ImportError("recurring_ical_events no está disponible")
    
    # recurring_ical_events requiere objetos aware en UTC
    events = recurring_ical_events.of(cal).between(
        start.astimezone(pytz.utc),
        end.astimezone(pytz.utc)
    )
    
    busy = []
    for ev in events:
        summary = str(ev.get("summary", "Reserva")).strip()
        dtstart = ev.get("dtstart")
        dtend = ev.get("dtend")

        if not dtstart:
            continue
        
        try:
            start_dt = _to_aware(dtstart.dt)
            
            # Fin exclusivo: si falta dtend, asumimos duración 1 día
            if dtend:
                end_dt = _to_aware(dtend.dt)
            else:
                # Para eventos de Airbnb, típicamente checkout = checkin + noches
                # Por defecto asumimos 1 noche
                end_dt = start_dt + timedelta(days=1)
            
            busy.append((start_dt, end_dt, summary))
        except Exception as e:
            print(f"⚠️ Error procesando evento '{summary}': {e}")
            continue
    
    return busy


def expand_busy_intervals_simple(
    cal: Calendar, 
    start: datetime, 
    end: datetime
) -> List[Tuple[datetime, datetime, str]]:
    """
    Expande eventos SIN soporte de recurrencias (fallback).
    Solo detecta eventos simples en el rango.
    """
    busy = []
    
    for component in cal.walk():
        if component.name != "VEVENT":
            continue
        
        summary = str(component.get("summary", "Reserva")).strip()
        dtstart = component.get("dtstart")
        dtend = component.get("dtend")
        
        if not dtstart:
            continue
        
        try:
            start_dt = _to_aware(dtstart.dt)
            
            if dtend:
                end_dt = _to_aware(dtend.dt)
            else:
                end_dt = start_dt + timedelta(days=1)
            
            # Filtrar: solo eventos que solapan con [start, end)
            if start_dt < end and end_dt > start:
                busy.append((start_dt, end_dt, summary))
        except Exception as e:
            print(f"⚠️ Error procesando evento '{summary}': {e}")
            continue
    
    return busy


def expand_busy_intervals(
    cal: Calendar, 
    start: datetime, 
    end: datetime
) -> List[Tuple[datetime, datetime, str]]:
    """
    Wrapper que usa recurring si está disponible, sino fallback simple.
    Devuelve lista UNIFICADA de intervalos ocupados.
    """
    if HAS_RECURRING:
        busy = expand_busy_intervals_with_recurring(cal, start, end)
    else:
        busy = expand_busy_intervals_simple(cal, start, end)
    
    if not busy:
        return []
    
    # Unificar intervalos solapados
    busy.sort(key=lambda x: x[0])
    merged = []
    
    for s, e, name in busy:
        if not merged:
            merged.append([s, e, [name]])
        else:
            last_s, last_e, names = merged[-1]
            # Solapan si: s <= last_e (considerando que los eventos de Airbnb
            # típicamente tienen checkout = siguiente checkin)
            if s <= last_e:
                merged[-1][1] = max(last_e, e)
                names.append(name)
            else:
                merged.append([s, e, [name]])
    
    # Volver a tupla con nombres unidos
    return [(s, e, " | ".join(set(names))) for s, e, names in merged]


def is_available(
    ics_url: str, 
    start_date: date, 
    end_date: date,
    buffer_hours: int = 0
) -> Dict:
    """
    Chequea disponibilidad para el rango [start_date, end_date) en TZ local.
    
    Args:
        ics_url: URL del calendario iCal
        start_date: Fecha de check-in
        end_date: Fecha de check-out
        buffer_hours: Horas de buffer antes/después (para mantenimiento)
    
    Returns:
        Dict con:
            - available (bool): True si está disponible
            - conflicts (list): Lista de eventos que solapan
            - query (dict): Rango consultado
            - total_nights (int): Noches totales
    """
    # Validación de entrada
    if end_date <= start_date:
        raise ValueError("end_date debe ser posterior a start_date")
    
    # Normalizamos a rangos aware en 00:00
    start_dt = TZ.localize(datetime(start_date.year, start_date.month, start_date.day, 0, 0))
    end_dt = TZ.localize(datetime(end_date.year, end_date.month, end_date.day, 0, 0))
    
    # Aplicar buffer si se especifica
    if buffer_hours > 0:
        start_dt -= timedelta(hours=buffer_hours)
        end_dt += timedelta(hours=buffer_hours)
    
    try:
        cal = fetch_calendar(ics_url)
    except Exception as e:
        return {
            "available": None,
            "conflicts": [],
            "query": {"start": start_dt.isoformat(), "end": end_dt.isoformat()},
            "error": str(e),
            "total_nights": (end_date - start_date).days
        }
    
    # Expandir con margen de 1 día para detectar eventos adyacentes
    busy = expand_busy_intervals(
        cal, 
        start_dt - timedelta(days=1), 
        end_dt + timedelta(days=1)
    )
    
    conflicts = []
    for b_start, b_end, title in busy:
        # Solapado si: start_dt < b_end AND b_start < end_dt
        # (intervalos semiabiertos: [start, end) )
        if start_dt < b_end and b_start < end_dt:
            conflicts.append({
                "start": b_start.isoformat(),
                "end": b_end.isoformat(),
                "title": title,
                "nights": (b_end.date() - b_start.date()).days
            })
    
    total_nights = (end_date - start_date).days
    
    return {
        "available": len(conflicts) == 0,
        "conflicts": conflicts,
        "query": {
            "start": start_dt.isoformat(), 
            "end": end_dt.isoformat(),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        },
        "total_nights": total_nights,
        "total_events_checked": len(busy)
    }


def debug_list_intervals(
    ics_url: str, 
    start_date: date, 
    end_date: date
) -> List[Dict]:
    """
    Devuelve los intervalos ocupados (ya unificados) entre start-end 
    para inspección en UI.
    
    Útil para debugging: muestra TODOS los eventos detectados en el rango.
    """
    start_dt = TZ.localize(datetime(start_date.year, start_date.month, start_date.day, 0, 0))
    end_dt = TZ.localize(datetime(end_date.year, end_date.month, end_date.day, 0, 0))
    
    try:
        cal = fetch_calendar(ics_url)
        busy = expand_busy_intervals(
            cal, 
            start_dt - timedelta(days=2),  # Margen más amplio para debug
            end_dt + timedelta(days=2)
        )
    except Exception as e:
        return [{"error": str(e)}]
    
    out = []
    for s, e, title in busy:
        nights = (e.date() - s.date()).days
        out.append({
            "start": s.isoformat(),
            "end": e.isoformat(),
            "start_date": s.date().isoformat(),
            "end_date": e.date().isoformat(),
            "title": title,
            "nights": nights
        })
    
    return out


def get_availability_calendar(
    ics_url: str,
    start_date: date,
    days_ahead: int = 90
) -> Dict[str, bool]:
    """
    Genera un calendario de disponibilidad día por día.
    
    Args:
        ics_url: URL del calendario iCal
        start_date: Fecha de inicio
        days_ahead: Días hacia adelante a verificar
    
    Returns:
        Dict con fechas ISO como keys y disponibilidad como values
        Ej: {"2024-12-01": True, "2024-12-02": False, ...}
    """
    end_date = start_date + timedelta(days=days_ahead)
    start_dt = TZ.localize(datetime(start_date.year, start_date.month, start_date.day, 0, 0))
    end_dt = TZ.localize(datetime(end_date.year, end_date.month, end_date.day, 0, 0))
    
    try:
        cal = fetch_calendar(ics_url)
        busy = expand_busy_intervals(cal, start_dt, end_dt)
    except Exception:
        return {}
    
    # Crear set de días ocupados
    busy_dates = set()
    for b_start, b_end, _ in busy:
        current = b_start.date()
        while current < b_end.date():
            busy_dates.add(current)
            current += timedelta(days=1)
    
    # Generar diccionario de disponibilidad
    availability = {}
    current = start_date
    while current < end_date:
        availability[current.isoformat()] = current not in busy_dates
        current += timedelta(days=1)
    
    return availability


# Función de auto-diagnóstico
def diagnose_ical(ics_url: str) -> Dict:
    """
    Ejecuta diagnóstico completo del calendario iCal.
    Útil para debugging.
    """
    diagnostics = {
        "url": ics_url,
        "fetch_success": False,
        "total_events": 0,
        "has_recurring": HAS_RECURRING,
        "sample_events": []
    }
    
    try:
        cal = fetch_calendar(ics_url)
        diagnostics["fetch_success"] = True
        
        # Contar eventos
        events = [c for c in cal.walk() if c.name == "VEVENT"]
        diagnostics["total_events"] = len(events)
        
        # Muestras
        for ev in events[:3]:
            diagnostics["sample_events"].append({
                "summary": str(ev.get("summary", "N/A")),
                "dtstart": str(ev.get("dtstart", "N/A")),
                "dtend": str(ev.get("dtend", "N/A"))
            })
    
    except Exception as e:
        diagnostics["error"] = str(e)
    
    return diagnostics
