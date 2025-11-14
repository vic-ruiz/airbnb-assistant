# check_ical_demo.py
from datetime import date
from dotenv import load_dotenv
import os
from ical_utils import is_available, debug_list_intervals

load_dotenv()

ICAL = os.environ.get("ICAL_RECOLETA")
if not ICAL:
    raise SystemExit("Falta ICAL_RECOLETA en .env")

# 1) Listar eventos que se leen del .ics en el mes
print("Eventos leídos (dic 2025):")
for ev in debug_list_intervals(ICAL, date(2025,12,1), date(2025,12,31)):
    print("-", ev)

# 2) Probar una consulta concreta (cambiá las fechas a gusto)
print("\nConsulta disponibilidad 15→18 dic 2025:")
print(is_available(ICAL, date(2025,12,15), date(2025,12,18)))

