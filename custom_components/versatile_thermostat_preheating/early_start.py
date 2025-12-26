from __future__ import annotations
from datetime import timedelta
import logging
import traceback
from homeassistant.core import HomeAssistant
from homeassistant.helpers.event import async_track_point_in_utc_time, async_track_state_change_event
from homeassistant.util import dt as dt_util

from homeassistant.core import callback
from homeassistant.helpers.event import (
    async_track_state_change_event,
    async_track_point_in_utc_time,
    async_call_later,
)
from .const import *

_LOGGER = logging.getLogger(__name__)

import re
def _ensure_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]

def _normalize_action(act) -> dict:
    """Retourne toujours un dict. Gère str→dict via literal_eval, puis fallback regex."""
    if isinstance(act, dict):
        return act
    if isinstance(act, str):
        s = act.strip()
        # 1) format Python dict en str -> literal_eval
        if s.startswith("{") and s.endswith("}"):
            try:
                d = ast.literal_eval(s)
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
        # 2) fallback: extraire 'service' et paires k:v très simples
        d: dict = {}
        m = re.search(r"\b([a-z_]+)[./]([a-z_]+)\b", s, re.I)
        if m:
            d["service"] = f"{m.group(1).lower()}.{m.group(2).lower()}"
        # paires "key: value"
        for k, v in re.findall(r"([a-z_]+)\s*:\s*([^\s,]+)", s, re.I):
            k = k.lower()
            if k in ("entity_id", "entities"):
                d["entity_id"] = [x.strip().lower() for x in v.split(",")]
            elif k in ("temperature", "target_temp", "setpoint"):
                try:
                    d[k] = float(v)
                except Exception:
                    pass
            else:
                d[k] = v
        return d
    return {}

def _is_climate_set_temp_action_dict(act: dict) -> bool:
    svc = act.get("service") or act.get("action")
    return isinstance(svc, str) and svc.replace("/", ".") == "climate.set_temperature"

def _extract_entities_from_action_dict(act: dict) -> set[str]:
    ents: set[str] = set()
    for key in ("entity_id", "entities"):
        v = act.get(key)
        for e in _ensure_list(v):
            if isinstance(e, str):
                ents.add(e.lower())
    for cont_key in ("service_data", "data"):
        cont = act.get(cont_key)
        if isinstance(cont, dict):
            v = cont.get("entity_id")
            for e in _ensure_list(v):
                if isinstance(e, str):
                    ents.add(e.lower())
    return ents

def _extract_temp_from_action_dict(act: dict) -> float | None:
    for cont_key in ("service_data", "data"):
        cont = act.get(cont_key)
        if isinstance(cont, dict):
            for k in ("temperature", "target_temp", "setpoint"):
                if k in cont:
                    try:
                        return float(cont[k])
                    except Exception:
                        pass
    for k in ("temperature", "target_temp", "setpoint"):
        if k in act:
            try:
                return float(act[k])
            except Exception:
                pass
    return None

class EarlyStart:

    def _compute_lead_minutes(self, cur_room_temp, target_at_trigger):
        """
        Calcule le nombre de minutes d'avance nécessaires pour atteindre la température cible,
        en utilisant le coefficient de chauffe heat_rate_coef (°C/h) si défini sur vt.
        """
        heat_rate_coef = getattr(self.vt, "heat_rate_coef", None)
        try:
            rate = float(heat_rate_coef) if heat_rate_coef is not None else 1.0
        except Exception:
            rate = 1.0
        if rate <= 0:
            rate = 1.0
        delta = target_at_trigger - cur_room_temp
        if delta <= 0:
            return 0
        # lead time en minutes = (delta °C) / (rate °C/h) * 60
        return int((delta / rate) * 60)
        
    def __init__(self, hass, vt_entity):
        self.hass = hass
        self.vt = vt_entity
        self.entry = vt_entity._entry
        self._unsub_timer = None
        self._unsubs = []
        self._next_fire_utc = None
        self._cached_next_temp = None  # <— on garde la prochaine temp du scheduler

    async def async_init(self):
        _LOGGER.debug("%s - EarlyStart.async_init() for %s", self.vt, self.vt.entity_id)

        # climate → callback ASYNC accepté
        self._unsubs.append(
            async_track_state_change_event(self.hass, [self.vt.entity_id], self._on_any_state_change)
        )

        sched_cfg = self._opt(OPT_EARLY_SCHED)

        # Normalisation en liste
        if not sched_cfg:
            sched_ids = []
        elif isinstance(sched_cfg, str):
            sched_ids = [sched_cfg]
        else:
            # On suppose que c'est déjà un iterable de strings (liste/tuple)
            sched_ids = [e for e in sched_cfg if e]

        if sched_ids:
            self._unsubs.append(
                async_track_state_change_event(
                    self.hass,
                    sched_ids,              # 👈 plusieurs entités possibles ici
                    self._on_any_state_change,
                )
            )

        out_id = self._opt(OPT_OUTDOOR)
        if out_id:
            self._unsubs.append(
                async_track_state_change_event(self.hass, [out_id], self._on_any_state_change)
            )

        # premier calcul différé : callback synchrone décoré @callback
        async_call_later(self.hass, 1.0, self._initial_later_cb)

    @callback
    def _initial_later_cb(self, _now):
        # on est sur le thread event loop → safe d’enfiler une tâche
        self.hass.async_create_task(self.async_reschedule())

    async def _on_any_state_change(self, event):
        # callback ASYNC → on peut await
        await self.async_reschedule()

    # ---------- helpers config ----------
    def _opt(self, key, default=None):
        # Chez toi, le flow écrit dans entry.data (pas options)
        try:
            return self.entry.options.get(key, self.entry.data.get(key, default))
        except Exception:
            return self.entry.data.get(key, default)

    # ---------- lecture scheduler ----------
    def _read_next_from_scheduler(self):
        """
        Retourne (next_trigger_utc, next_target_temp) à partir du/ des scheduler(s).
        Compatible Scheduler (nielsfaber) :
        - attributes.next_trigger : ISO
        - attributes.next_slot : index du prochain timeslot
        - attributes.timeslots[next_slot].actions : liste d'index dans attributes.actions
        - attributes.actions[i] : dict {service, entity_id(s), ... , temperature}
        """
        _LOGGER.debug("%s - EarlyStart_read_next_from_scheduler: Enter _read_next_from_scheduler", self.vt)

        # Normalisation : option = string (ancien format) ou liste (nouveau format multiple)
        sched_cfg = self._opt(OPT_EARLY_SCHED)
        _LOGGER.debug("%s - EarlyStart_read_next_from_scheduler: raw scheduler option = %s", self.vt, sched_cfg)

        if not sched_cfg:
            return None, None

        if isinstance(sched_cfg, str):
            sched_ids = [sched_cfg]
        else:
            # On essaie d'itérer dessus, et on filtre les valeurs vides
            try:
                sched_ids = [s for s in sched_cfg if s]
            except TypeError:
                # Au cas où un type exotique serait stocké
                sched_ids = [str(sched_cfg)]

        _LOGGER.debug("%s - EarlyStart_read_next_from_scheduler: normalized scheduler ids = %s", self.vt, sched_ids)

        best_dt_utc: datetime | None = None
        best_temp: float | None = None
        best_sched_id: str | None = None

        # Parcourt tous les schedulers sélectionnés
        for sched_id in sched_ids:
            st = self.hass.states.get(sched_id)
            _LOGGER.debug("%s - EarlyStart_read_next_from_scheduler: checking scheduler %s -> %s", self.vt, sched_id, st)

            if not st:
                _LOGGER.debug("%s - EarlyStart_read_next_from_scheduler: scheduler %s not found", self.vt, sched_id)
                continue

            attrs = st.attributes or {}

            # 1) Heure du prochain changement
            ts = attrs.get("next_trigger")
            _LOGGER.debug("%s - EarlyStart_read_next_from_scheduler(%s): next_trigger attr = %s", self.vt, sched_id, ts)

            if ts is None and isinstance(st.state, str) and "T" in st.state:
                ts = st.state

            _LOGGER.debug("%s - EarlyStart_read_next_from_scheduler(%s): effective ts = %s", self.vt, sched_id, ts)

            next_dt_utc = None
            if isinstance(ts, str):
                try:
                    _dt = dt_util.parse_datetime(ts)
                    if _dt:
                        next_dt_utc = dt_util.as_utc(_dt)
                except Exception:
                    _LOGGER.exception("%s - EarlyStart_read_next_from_scheduler(%s): error parsing next_trigger", self.vt, sched_id)

            _LOGGER.debug("%s - EarlyStart_read_next_from_scheduler(%s): parsed next_dt_utc = %s", self.vt, sched_id, next_dt_utc)

            next_temp = None

            # 2) Tentative 1 : lecture directe (au cas où le scheduler expose déjà la temp)
            if attrs:
                for k in ("next_setpoint", "next_temperature", "next_target", "target", "setpoint", "temperature"):
                    v = attrs.get(k)
                    if v is None:
                        continue
                    try:
                        if isinstance(v, dict) and "temperature" in v:
                            v = v["temperature"]
                        next_temp = float(v)
                        break
                    except Exception:
                        continue

            # 3) Scheduler: si pas de temp directe, déduire via next_slot/actions
            if next_temp is None:
                try:
                    next_slot = attrs.get("next_slot")
                    timeslots = attrs.get("timeslots") or []
                    actions = attrs.get("actions") or []

                    # next_slot peut être str
                    if isinstance(next_slot, str) and next_slot.isdigit():
                        next_slot = int(next_slot)

                    # 3.a) Essayer actions indexées dans le slot (si le slot est un dict)
                    slot = None
                    if isinstance(timeslots, list) and isinstance(next_slot, int) and 0 <= next_slot < len(timeslots):
                        slot = timeslots[next_slot]
                    elif isinstance(timeslots, dict) and str(next_slot) in timeslots:
                        slot = timeslots[str(next_slot)]

                    # cibles acceptées (climate + underlyings + attrs.entities)
                    target_ents: set[str] = {self.vt.entity_id.lower()}
                    try:
                        for e in getattr(self.vt, "underlying_entities", []):
                            if isinstance(e, str):
                                target_ents.add(e.lower())
                    except Exception:
                        pass
                    ents_attr = attrs.get("entities")
                    if isinstance(ents_attr, list):
                        for e in ents_attr:
                            if isinstance(e, str):
                                target_ents.add(e.lower())

                    # i) si le slot expose des actions (inline ou indexes)
                    got_from_slot = False
                    if isinstance(slot, dict) and "actions" in slot:
                        slot_actions = slot.get("actions")
                        idx_list: list[int] = []
                        inline_actions: list = []
                        if isinstance(slot_actions, int):
                            idx_list = [slot_actions]
                        elif isinstance(slot_actions, list):
                            for it in slot_actions:
                                if isinstance(it, int):
                                    idx_list.append(it)
                                else:
                                    inline_actions.append(it)
                        elif isinstance(slot_actions, (dict, str)):
                            inline_actions.append(slot_actions)

                        # Parcours des actions référencées
                        for idx in idx_list:
                            if isinstance(idx, int) and 0 <= idx < len(actions):
                                act = _normalize_action(actions[idx])
                                if not act:
                                    continue
                                if _is_climate_set_temp_action_dict(act):
                                    act_ents = _extract_entities_from_action_dict(act)
                                    if not act_ents or (act_ents & target_ents):
                                        t = _extract_temp_from_action_dict(act)
                                        if t is not None:
                                            next_temp = t
                                            got_from_slot = True
                                            break
                        # Parcours des inline
                        if not got_from_slot:
                            for raw in inline_actions:
                                act = _normalize_action(raw)
                                if not act:
                                    continue
                                if _is_climate_set_temp_action_dict(act):
                                    act_ents = _extract_entities_from_action_dict(act)
                                    if not act_ents or (act_ents & target_ents):
                                        t = _extract_temp_from_action_dict(act)
                                        if t is not None:
                                            next_temp = t
                                            got_from_slot = True
                                            break

                    # ii) sinon: fallback → prendre l'action à l'index next_slot
                    if not got_from_slot and isinstance(next_slot, int) and 0 <= next_slot < len(actions):
                        act = _normalize_action(actions[next_slot])
                        if _is_climate_set_temp_action_dict(act):
                            act_ents = _extract_entities_from_action_dict(act)
                            # si pas d'entité dans l'action, on tolère et on s'appuie sur attrs.entities
                            if not act_ents or (act_ents & target_ents):
                                t = _extract_temp_from_action_dict(act)
                                if t is not None:
                                    next_temp = t

                    # Debug si toujours rien
                    if next_temp is None:
                        _LOGGER.debug(
                            "%s - EarlyStart_read_next_from_scheduler(%s): next_slot=%s slot=%s; picked_action=%s",
                            self.vt,
                            sched_id,
                            next_slot,
                            str(slot)[:200] if slot is not None else None,
                            str(actions[next_slot])[:200] if isinstance(next_slot, int) and 0 <= next_slot < len(actions) else None,
                        )

                except Exception:
                    _LOGGER.exception(
                        "%s - EarlyStart_read_next_from_scheduler(%s): Exception while parsing scheduler (timeslots/actions)",
                        self.vt,
                        sched_id,
                    )

            # Si ce scheduler n'a pas de temps ou pas de température, on l'ignore
            if next_dt_utc is None or next_temp is None:
                _LOGGER.debug(
                    "%s - EarlyStart_read_next_from_scheduler(%s): incomplete -> next_time=%s next_temp=%s (keys=%s)",
                    self.vt,
                    sched_id,
                    next_dt_utc,
                    next_temp,
                    list(attrs.keys()),
                )
                continue

            # Sélectionne le scheduler avec le next_trigger le plus proche
            if best_dt_utc is None or next_dt_utc < best_dt_utc:
                best_dt_utc = next_dt_utc
                best_temp = next_temp
                best_sched_id = sched_id

        _LOGGER.debug(
            "%s - EarlyStart_read_next_from_scheduler: chosen scheduler=%s next_dt_utc=%s next_temp=%s",
            self.vt,
            best_sched_id,
            best_dt_utc,
            best_temp,
        )

        return best_dt_utc, best_temp

    def _current_target_temperature(self):
        """Consigne actuelle (target) lue depuis la Climate ou son état."""
        # 1) Propriété ClimateEntity si dispo
        t = getattr(self.vt, "target_temperature", None)
        if t is not None:
            return t

        # 2) Depuis l'état Home Assistant
        st = self.hass.states.get(self.vt.entity_id)
        if st:
            # HA stocke souvent la consigne sous 'temperature'
            for k in ("temperature", "target_temperature"):
                v = st.attributes.get(k)
                if v is not None:
                    try:
                        return float(v)
                    except Exception:
                        pass

        # 3) Fallback depuis tes attributs custom (vus dans tes logs)
        attrs = getattr(self.vt, "extra_state_attributes", {}) or {}
        for k in ("regulated_target_temperature", "saved_target_temp"):
            v = attrs.get(k)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
        return None

    def _target_from_mode(self, next_sched_temp: float | None):
        """
        Calcule la cible à atteindre à l'heure du scheduler selon le mode choisi.
        - auto_from_scheduler : utilise next_sched_temp
        - fixed_temperature  : options fixe
        - comfort_preset     : temp comfort exposée par la climate / attributs
        """
        mode = self._opt(OPT_EARLY_MODE, "comfort_preset")

        if mode == "auto_from_scheduler":
            return next_sched_temp

        if mode == "fixed_temperature":
            return self._opt(OPT_EARLY_TEMP)

        # comfort_preset
        t = getattr(self.vt, "comfort_temperature", None)
        if t is not None:
            return t
        attrs = getattr(self.vt, "extra_state_attributes", {}) or {}
        v = attrs.get("comfort_temp")
        try:
            return float(v) if v is not None else None
        except Exception:
            return None

    # ---------- planification ----------
    async def async_reschedule(self):
        if self._unsub_timer:
            self._unsub_timer(); self._unsub_timer = None

        if not self._opt(OPT_EARLY_ENABLED, False):
            return self._publish_debug(None, None, None)

        next_dt_utc, next_sched_temp = self._read_next_from_scheduler()
        if not next_dt_utc:
            _LOGGER.debug("%s - EarlyStart: no next_trigger found", self.vt)
            return self._publish_debug(None, None, None)
        
        _LOGGER.debug("%s - EarlyStart_read_next_from_scheduler temp=%s at=%s", self.vt, next_sched_temp, next_dt_utc)

        self._cached_next_temp = next_sched_temp  # utilisable à l'exécution

        cur_room_temp = getattr(self.vt, "current_temperature", None)
        cur_target = self._current_target_temperature()
        target_at_trigger = self._target_from_mode(next_sched_temp)

        if cur_room_temp is None or target_at_trigger is None:
            _LOGGER.debug("%s - EarlyStart: missing temperatures cur=%s target=%s", self.vt, cur_room_temp, target_at_trigger)
            return self._publish_debug(None, None, None)

        # 👉 Tu veux anticiper SEULEMENT si la température du prochain scheduler
        #    est supérieure à la consigne actuelle.
        _LOGGER.debug("%s - EarlyStart: OPT_EARLY_MODE=%s cur_target=%f nextTemp=%f", self.vt, self._opt(OPT_EARLY_MODE, "comfort_preset"), cur_target, next_sched_temp)
        if self._opt(OPT_EARLY_MODE, "comfort_preset") == "auto_from_scheduler":
            if cur_target is not None and next_sched_temp is not None:
                tol = float(self._opt(OPT_HEAT_TOLERANCE, 0.2))
                if next_sched_temp <= (cur_target + tol):
                    _LOGGER.debug(
                        " %s - EarlyStart: next_sched_temp %.2f ≤ current target %.2f (+tol %.2f) -> no preheat",
                        self.vt, next_sched_temp, cur_target, tol
                    )
                    return self._publish_debug(None, 0, None)

        # ✅ Anticiper uniquement si besoin de chauffer (room temp < target)
        if not self._needs_heating(cur_room_temp, target_at_trigger):
            _LOGGER.debug("%s - EarlyStart: no heating need (cur=%.2f target=%.2f at %s)", self.vt, cur_room_temp, target_at_trigger, next_dt_utc)
            return self._publish_debug(None, 0, None)

        heat_rate_coef = getattr(self.vt, "heat_rate_coef", None)
        lead_min = self._compute_lead_minutes(cur_room_temp, target_at_trigger)
        _LOGGER.debug("%s - EarlyStart: lead_min=%f heat_rate_coef=%s)", self.vt, lead_min, heat_rate_coef)

        fire_at = next_dt_utc - timedelta(minutes=lead_min)
        now = dt_util.utcnow()
        # S'assurer que les deux datetimes sont offset-aware (UTC)
        if fire_at.tzinfo is None:
            fire_at = fire_at.replace(tzinfo=dt_util.UTC)
        if now.tzinfo is None:
            now = now.replace(tzinfo=dt_util.UTC)

        if fire_at <= now:
            _LOGGER.debug("%s - EarlyStart: fire_at in the past -> skip (lead=%s)", self.vt, lead_min)
            return self._publish_debug(None, lead_min, heat_rate_coef)

        from homeassistant.helpers.event import async_track_point_in_utc_time
        @callback
        def _timer_cb(_):
            # on est *dans* l’event loop → safe
            self.hass.async_create_task(self._apply_preheat_action())

        self._unsub_timer = async_track_point_in_utc_time(self.hass, _timer_cb, fire_at)
        self._publish_debug(fire_at, lead_min, heat_rate_coef)
        _LOGGER.debug("%s - EarlyStart: scheduled at %s (lead %s min, target_at_trigger=%.2f)", self.vt, fire_at, lead_min, target_at_trigger)

    async def _fire_cb(self, _):
        await self._apply_preheat_action()

    async def _apply_preheat_action(self):
        """À l'heure planifiée, applique la bonne consigne."""
        mode = self._opt(OPT_EARLY_MODE, "comfort_preset")

        if mode == "auto_from_scheduler":
            temp = self._cached_next_temp
            if temp is not None:
                await self.hass.services.async_call(
                    "climate", "set_temperature",
                    {"entity_id": self.vt.entity_id, "temperature": float(temp)},
                    blocking=False,
                )
                return

        if mode == "fixed_temperature":
            temp = self._opt(OPT_EARLY_TEMP)
            if temp is not None:
                await self.hass.services.async_call(
                    "climate", "set_temperature",
                    {"entity_id": self.vt.entity_id, "temperature": float(temp)},
                    blocking=False,
                )
                return

        # défaut : preset comfort
        await self.hass.services.async_call(
            "climate", "set_preset_mode",
            {"entity_id": self.vt.entity_id, "preset_mode": "comfort"},
            blocking=False,
        )

    def _needs_heating(self, cur, target) -> bool:
        if not self._opt(OPT_ONLY_IF_HEATING, True):
            return True
        tol = float(self._opt(OPT_HEAT_TOLERANCE, 0.2))
        if (target - cur) <= tol:
            return False
        hvac_mode = getattr(self.vt, "hvac_mode", None)
        if hvac_mode in ("off", "cool", "dry", "fan_only"):
            return False
        if getattr(self.vt, "window_open", False):
            return False
        return True

    def _publish_debug(self, fire_at, lead_minutes, heat_rate):
        self.vt.early_start_info = {
            "next_fire_utc": fire_at.isoformat() if fire_at else None,
            "lead_minutes": lead_minutes,
            "heat_rate_c_per_h": heat_rate,
        }
        if hasattr(self.vt, "async_write_ha_state"):
            self.vt.async_write_ha_state()
