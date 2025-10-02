#!/usr/bin/env python3
# automatic_regulator_full.py
"""
Complete integrated script: Phases 1-3 (features, replay/trainer, hybrid mode).
Skip: Phase 4 (personalization/weather).
"""

import os
import sys
import time
import json
import math
import random
import logging
import threading
from collections import deque, defaultdict
from datetime import datetime

# ML libs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Kafka (producer + consumer)
from confluent_kafka import Producer, Consumer, KafkaError

# ============== Logging Setup =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("fan_regulator.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger("FanRegulator")

# ================= Phase-1: Feature Schema & Helpers =================
# Master feature order (11 features)
FEATURE_NAMES = [
    "indoor_temperature",      # °C
    "outdoor_temperature",     # °C
    "indoor_humidity",         # %
    "outdoor_humidity",        # %
    "hour_of_day",             # 0-23
    "day_of_week",             # 0-6
    "season_code",             # 0..3
    "previous_fan_level",      # 1..5
    "recent_adjustment_trend", # -1,0,1  (down,stable,up)
    "temperature_diff",        # indoor - outdoor
    "humidity_index"           # indoor_humidity - outdoor_humidity
]
INPUT_DIM = len(FEATURE_NAMES)  # 11

_SEASON_MAP = {"winter": 0, "summer": 1, "monsoon": 2, "autumn": 3}
_TREND_MAP = {"down": -1, "stable": 0, "steady": 0, "none": 0, "up": 1}

def _coerce_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

def _coerce_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return int(default)

def _season_to_code(season_str: str) -> int:
    if not isinstance(season_str, str):
        return 0
    return _SEASON_MAP.get(season_str.lower(), 0)

def _trend_to_code(trend_str: str) -> int:
    if not isinstance(trend_str, str):
        return 0
    return _TREND_MAP.get(trend_str.lower(), 0)

def _infer_season_from_month(month: int) -> str:
    if month in (12, 1, 2):
        return "winter"
    elif month in (3, 4, 5):
        return "summer"
    elif month in (6, 7, 8, 9):
        return "monsoon"
    else:
        return "autumn"

def extract_features(data: dict) -> list:
    """
    Build the 11-D feature vector in FEATURE_NAMES order.
    Works even if some fields are missing; uses defaults/derivations.
    """
    indoor_t  = _coerce_float(data.get("indoor_temperature", 25.0))
    outdoor_t = _coerce_float(data.get("outdoor_temperature", 28.0))
    indoor_h  = _coerce_float(data.get("indoor_humidity", 55.0))
    outdoor_h = _coerce_float(data.get("outdoor_humidity", 60.0))

    hour      = _coerce_int(data.get("hour_of_day", time.localtime().tm_hour))
    dow       = _coerce_int(data.get("day_of_week", time.localtime().tm_wday))

    season_str = data.get("season")
    if not season_str:
        season_str = _infer_season_from_month(time.localtime().tm_mon)
    season_code = _season_to_code(season_str)

    prev_level = _coerce_int(data.get("previous_fan_level", 2))
    trend_code = _trend_to_code(data.get("recent_adjustment_trend", "stable"))

    temp_diff      = round(indoor_t - outdoor_t, 3)
    humidity_index = round(indoor_h - outdoor_h, 3)

    feat = [
        indoor_t,
        outdoor_t,
        indoor_h,
        outdoor_h,
        hour,
        dow,
        season_code,
        prev_level,
        trend_code,
        temp_diff,
        humidity_index
    ]
    return feat

def sanitize_target(level) -> int:
    lv = _coerce_int(level, 3)
    return max(1, min(5, lv))

def clamp_fan_level(x: float) -> int:
    return int(max(1, min(5, round(x))))

# ================= Phase-1 Helper: bounded replay buffer creator =================
def create_replay_buffer(max_size=1000):
    return deque(maxlen=max_size)

# ================= Phase-2: Smart Loss + ReplayBuffer + Trainer =================

# --- Smart loss (Huber + penalty for large errors) ---
class SmartLoss(nn.Module):
    """
    Combines SmoothL1 (Huber) + quadratic penalty for large mistakes beyond margin.
    """
    def __init__(self, margin=1.0, alpha=0.25):
        super().__init__()
        self.huber = nn.SmoothL1Loss()
        self.margin = margin
        self.alpha = alpha

    def forward(self, pred, target):
        base = self.huber(pred, target)
        diff = torch.abs(pred - target)
        penalty = torch.clamp(diff - self.margin, min=0.0) ** 2
        return base + self.alpha * penalty.mean()

def create_smart_loss(margin=1.0, alpha=0.25):
    return SmartLoss(margin=margin, alpha=alpha)

# --- ReplayBuffer class (Phase-2) ---
class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buf = deque(maxlen=max_size)

    def add(self, features, target):
        self.buf.append((features, target))

    def __len__(self):
        return len(self.buf)

    def sample(self, batch_size):
        k = min(batch_size, len(self.buf))
        if k <= 0:
            return []
        return random.sample(self.buf, k)

# --- Scheduler factory ---
def create_scheduler(optimizer, mode="plateau"):
    if mode == "plateau":
        # return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=False)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    else:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

# --- Early stop (lightweight monitor) ---
class EarlyStopSignal:
    def __init__(self, patience=200):
        self.best = math.inf
        self.bad_steps = 0
        self.patience = patience

    def step(self, loss_value: float) -> bool:
        if loss_value + 1e-6 < self.best:
            self.best = loss_value
            self.bad_steps = 0
        else:
            self.bad_steps += 1
        return self.bad_steps >= self.patience

# --- BatchTrainer ---
class BatchTrainer:
    def __init__(self, model, optimizer, criterion, scheduler=None, clip_grad=1.0, device=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.clip_grad = clip_grad
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.early = EarlyStopSignal(patience=1000)

    def train_once(self, batch):
        if not batch:
            return None
        feats, targs = zip(*batch)
        X = torch.tensor(feats, dtype=torch.float32, device=self.device)
        y = torch.tensor(targs, dtype=torch.float32, device=self.device).unsqueeze(1)
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(X)
        loss = self.criterion(out, y)
        loss.backward()
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            try:
                self.scheduler.step(loss.item())
            except Exception:
                pass
        elif self.scheduler is not None:
            try:
                self.scheduler.step()
            except Exception:
                pass
        stop_flag = self.early.step(loss.item())
        return float(loss.item()), stop_flag

# ================= Model Definition =================
class FanRegulatorModel(nn.Module):
    """Neural network predicting fan level from sensor data."""
    def __init__(self, input_dim=INPUT_DIM):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # simple MLP
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        ).to(self.device)

    def forward(self, x):
        return self.net(x) * 4 + 1  # encourage output centered 1..5 (clamped later)

# ================= OnlineLearner (integrates Phase-1 & Phase-2) =================
class OnlineLearner:
    def __init__(self, model_path="fan_model.pt", input_dim=INPUT_DIM):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FanRegulatorModel(input_dim=input_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        # Phase-2 components
        self.replay = ReplayBuffer(max_size=10000)
        self.criterion = create_smart_loss(margin=1.0, alpha=0.25)
        self.scheduler = create_scheduler(self.optimizer, mode="plateau")
        self.trainer = BatchTrainer(
            model=self.model,
            optimizer=self.optimizer,
            criterion=self.criterion,
            scheduler=self.scheduler,
            clip_grad=1.0,
            device=self.device
        )

        self.feature_history = []
        self.update_interval_sec = 5.0
        self.min_samples_to_train = 64
        self.minibatch_size = 64
        self.last_update = time.time()
        self.model_version = 0

        # Try load existing
        if os.path.exists(self.model_path):
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info("✅ Loaded existing model successfully.")
            except RuntimeError as e:
                logger.warning(f"⚠️ Checkpoint mismatch: {e}. Resetting model...")
                try:
                    os.remove(self.model_path)
                except Exception:
                    pass
        else:
            logger.info("🆕 No checkpoint found. Starting fresh model.")

    def predict(self, features):
        self.model.eval()
        try:
            with torch.no_grad():
                x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
                pred = self.model(x).item()
            return clamp_fan_level(pred)
        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")
            return 3

    def add_training_sample(self, features, target):
        # store floats
        self.replay.add(features, float(target))
        self.feature_history.append(features)
        # Trigger update if conditions met
        if (time.time() - self.last_update) > self.update_interval_sec and len(self.replay) >= self.min_samples_to_train:
            self._update_model()

    def _update_model(self):
        try:
            batch = self.replay.sample(self.minibatch_size)
            result = self.trainer.train_once(batch)
            if result is None:
                return
            loss_value, _ = result
            logger.info(f"[Training] Loss={loss_value:.4f} | Version={self.model_version}")
            # save model state_dict
            try:
                torch.save(self.model.state_dict(), self.model_path)
            except Exception as e:
                logger.error(f"❌ Failed to save model: {e}")
            self.model_version += 1
            self.last_update = time.time()
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")

# ================= Phase-3: Multi-Mode Intelligence (Hybrid, Confidence, Safety) =================

# Mode history logger
class ModeHistory:
    def __init__(self, maxlen=500):
        self.events = deque(maxlen=maxlen)

    def log(self, from_mode: str, to_mode: str, reason: str = ""):
        evt = {"ts": time.time(), "from": from_mode.lower(), "to": to_mode.lower(), "reason": reason}
        self.events.append(evt)
        logger.info(f"🔀 Mode switch: {from_mode.upper()} → {to_mode.upper()} | {reason}")

# Confidence scorer using cosine similarity over replay buffer
class ConfidenceScorer:
    def __init__(self, replay_ref, k=50, tol=1.0):
        self.replay_ref = replay_ref
        self.k = k
        self.tol = tol

    @staticmethod
    def _cosine(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def score(self, features, pred_level: int) -> float:
        sims = []
        for f, t in getattr(self.replay_ref, "buf", []):
            sims.append((self._cosine(features, f), t))
        if not sims:
            return 0.0
        sims.sort(key=lambda x: x[0], reverse=True)
        top = sims[:min(self.k, len(sims))]
        if not top:
            return 0.0
        agree = sum(1 for _, t in top if abs(t - pred_level) <= self.tol)
        return float(agree) / len(top)

# Safety governor: EMA smoothing + max delta clamp
class SafetyGovernor:
    def __init__(self, max_delta_per_step=1, ema_alpha=0.6):
        self.max_delta = int(max_delta_per_step)
        self.alpha = float(ema_alpha)
        self._ema_state = None

    def _ema(self, x):
        if self._ema_state is None:
            self._ema_state = float(x)
        else:
            self._ema_state = self.alpha * float(x) + (1.0 - self.alpha) * self._ema_state
        return self._ema_state

    def apply(self, last_applied: int, suggested: int) -> int:
        smoothed = self._ema(suggested)
        delta = int(round(smoothed)) - int(last_applied)
        if delta > self.max_delta:
            return int(last_applied + self.max_delta)
        if delta < -self.max_delta:
            return int(last_applied - self.max_delta)
        return clamp_fan_level(int(round(smoothed)))

# Hybrid policy combining model pred + confidence + governor
class HybridPolicy:
    def __init__(self, learner, replay_ref, conf_threshold=0.7, max_delta_per_step=1, ema_alpha=0.6):
        self.learner = learner
        self.scorer = ConfidenceScorer(replay_ref=replay_ref, k=50, tol=1.0)
        self.governor = SafetyGovernor(max_delta_per_step=max_delta_per_step, ema_alpha=ema_alpha)
        self.conf_threshold = float(conf_threshold)

    def decide(self, features, last_applied: int):
        raw_pred = self.learner.predict(features)
        conf = self.scorer.score(features, raw_pred)
        logger.info(f"🤝 HYBRID: model_pred={raw_pred} | confidence={conf:.2f}")
        if conf >= self.conf_threshold:
            governed = self.governor.apply(last_applied, raw_pred)
            decision = {"mode": "auto", "final_level": governed, "suggested": raw_pred, "confidence": conf}
        else:
            decision = {"mode": "suggest", "final_level": last_applied, "suggested": raw_pred, "confidence": conf}
        return decision

# Voice command helper
class VoiceCommandProcessor:
    def __init__(self):
        try:
            import speech_recognition as sr  # optional dependency at runtime
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone() if hasattr(sr, "Microphone") else None
        except Exception:
            self.recognizer = None
            self.microphone = None
        self.voice_command = None

    def store_command(self, command):
        self.voice_command = command

    def get_command(self):
        cmd = self.voice_command
        self.voice_command = None
        return cmd

# ModeManager integrates manual/ai/voice/hybrid
class ModeManager:
    def __init__(self, learner):
        self.mode = "manual"
        self.learner = learner
        self.voice_processor = VoiceCommandProcessor()
        # neutral 11-D baseline for AI mode
        self.ai_features = [27.0, 30.0, 55.0, 60.0, 14, 2, 1, 2, 0, -3.0, -5.0]
        self.mode_history = ModeHistory(maxlen=500)
        self.last_applied_level = 3
        self.hybrid = HybridPolicy(
            learner=self.learner,
            replay_ref=self.learner.replay,
            conf_threshold=0.70,
            max_delta_per_step=1,
            ema_alpha=0.6
        )

    def set_mode(self, mode):
        mode = mode.lower()
        if mode in ["manual", "ai", "voice", "hybrid"]:
            if mode != self.mode:
                self.mode_history.log(self.mode, mode, reason="external request")
            self.mode = mode
            logger.info(f"🔀 Mode switched to: {self.mode.upper()}")
        else:
            logger.warning("⚠️ Invalid mode! Use: manual | ai | voice | hybrid")

    def handle_manual_mode(self, features, target):
        model_pred = self.learner.predict(features)
        self.learner.add_training_sample(features, target)
        self.last_applied_level = target
        return target, model_pred

    def handle_ai_mode(self, features=None):
        feats = features if features is not None else self.ai_features
        pred = self.learner.predict(feats)
        final = SafetyGovernor(max_delta_per_step=1, ema_alpha=0.6).apply(self.last_applied_level, pred)
        self.last_applied_level = final
        return final, pred

    def handle_voice_mode(self, features, target, voice_cmd=None):
        if voice_cmd:
            try:
                final = clamp_fan_level(int(voice_cmd))
            except Exception:
                final = target
            self.learner.add_training_sample(features, final)
            self.last_applied_level = final
            return final, self.learner.predict(features)
        else:
            pred = self.learner.predict(features)
            final = target
            self.last_applied_level = final
            return final, pred

    def handle_hybrid_mode(self, features):
        decision = self.hybrid.decide(features, self.last_applied_level)
        final = decision["final_level"]
        suggested = decision["suggested"]
        conf = decision["confidence"]
        mode_note = decision["mode"]
        logger.info(f"🧠 HYBRID decision: final={final} | suggested={suggested} | conf={conf:.2f} | mode={mode_note}")
        self.learner.add_training_sample(features, final)
        self.last_applied_level = final
        return final, suggested, conf

# ================= Kafka Consumer (integrates everything) =================
class KafkaFanConsumer:
    def __init__(self, topic, group_id, bootstrap_servers='localhost:9092', reset=False):
        self.topic = topic
        self.group_id = group_id
        self.bootstrap_servers = bootstrap_servers
        self.reset = reset

        self.conf = {
            'bootstrap.servers': self.bootstrap_servers,
            'group.id': self.group_id,
            'auto.offset.reset': 'earliest' if self.reset else 'latest',
            'enable.auto.commit': False,
            'fetch.wait.max.ms': 100,
            'fetch.min.bytes': 1,
            'max.partition.fetch.bytes': 1048576,
            'session.timeout.ms': 10000,
            'heartbeat.interval.ms': 3000,
            'isolation.level': 'read_committed'
        }

        self.consumer = Consumer(self.conf)
        self.learner = OnlineLearner()
        self.modes = ModeManager(self.learner)
        self.running = True
        self.last_heartbeat = time.time()

    def consume_messages(self):
        try:
            self.consumer.subscribe([self.topic])
            logger.info(f"🚀 Connected to Kafka topic: {self.topic}")
            while self.running:
                msg = self.consumer.poll(0.5)
                if msg is None:
                    continue
                if msg.error():
                    self._handle_error(msg.error())
                    continue
                try:
                    data = json.loads(msg.value().decode('utf-8'))
                    if isinstance(data, list) and len(data) > 0:
                        data = data[0]
                    features = extract_features(data)
                    target = sanitize_target(data.get('regulator_level', 3))
                    mode_in_message = str(data.get('mode', '')).lower()
                    if mode_in_message in ['manual', 'ai', 'voice', 'hybrid']:
                        self.modes.set_mode(mode_in_message)
                    # routing based on current mode
                    extra = ""
                    if self.modes.mode == "manual":
                        final, model_pred = self.modes.handle_manual_mode(features, target)
                    elif self.modes.mode == "ai":
                        final, model_pred = self.modes.handle_ai_mode(features)
                    elif self.modes.mode == "voice":
                        voice_cmd = data.get('voice_command')
                        final, model_pred = self.modes.handle_voice_mode(features, target, voice_cmd)
                        extra = f"| Voice={voice_cmd}"
                    elif self.modes.mode == "hybrid":
                        final, suggested, conf = self.modes.handle_hybrid_mode(features)
                        model_pred = suggested
                        extra = f"| Conf={conf:.2f}"
                    else:
                        final, model_pred = target, target

                    logger.info(
                        f"🎯 Target={target} | 🤖 Predicted={model_pred} | Final Fan Level={final} (Mode={self.modes.mode.upper()}) {extra}"
                    )
                    # commit offset to Kafka
                    try:
                        self.consumer.commit(message=msg)
                    except Exception:
                        pass
                except Exception as e:
                    logger.error(f"⚠️ Data issue: {e}")
        except KeyboardInterrupt:
            logger.info("🛑 Graceful shutdown requested...")
        finally:
            self.consumer.close()
            logger.info("✅ Consumer stopped.")

    def _handle_error(self, error):
        if error.code() == KafkaError._PARTITION_EOF:
            logger.warning("🔚 End of partition")
        else:
            logger.error(f"⚠️ Kafka error: {error}")

# ================= Kafka Producer Simulator (Phase-2 producer) =================
def get_season(month):
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "summer"
    elif month in [6, 7, 8, 9]:
        return "monsoon"
    else:
        return "autumn"

def simulate_sensor_data(prev_fan_level):
    indoor_temp = round(random.uniform(22, 32), 1)
    outdoor_temp = round(random.uniform(20, 38), 1)
    indoor_humidity = round(random.uniform(40, 70), 1)
    outdoor_humidity = round(random.uniform(30, 80), 1)
    now = datetime.now()
    hour_of_day = now.hour
    day_of_week = now.weekday()
    season = get_season(now.month)
    if indoor_temp > 30 or indoor_humidity > 65:
        regulator_level = random.choice([3, 4, 5])
    elif indoor_temp > 27:
        regulator_level = random.choice([2, 3, 4])
    else:
        regulator_level = random.choice([1, 2])
    if prev_fan_level is None:
        trend = "none"
    elif regulator_level > prev_fan_level:
        trend = "up"
    elif regulator_level < prev_fan_level:
        trend = "down"
    else:
        trend = "steady"
    message = {
        "indoor_temperature": indoor_temp,
        "outdoor_temperature": outdoor_temp,
        "indoor_humidity": indoor_humidity,
        "outdoor_humidity": outdoor_humidity,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "season": season,
        "previous_fan_level": prev_fan_level if prev_fan_level else 0,
        "recent_adjustment_trend": trend,
        "temperature_diff": round(indoor_temp - outdoor_temp, 3),
        "humidity_index": round(indoor_humidity - outdoor_humidity, 3),
        "regulator_level": regulator_level,
        "mode": "manuel"
    }
    return message, regulator_level

def run_producer_loop(brokers='localhost:9092', topic="fan_regulator_data", interval=2.0):
    conf = {'bootstrap.servers': brokers}
    producer = Producer(conf)
    prev_level = None
    logger.info("🚀 Starting Kafka Producer simulator...")
    try:
        while True:
            data, prev_level = simulate_sensor_data(prev_level)
            payload = json.dumps(data)
            producer.produce(topic, key="sensor", value=payload)
            producer.poll(0)
            logger.info(f"📡 Sent: {payload}")
            time.sleep(interval)
    except KeyboardInterrupt:
        logger.info("🛑 Producer stopped by user.")
    finally:
        try:
            producer.flush()
        except Exception:
            pass

# ================= Consumer launcher =================

def run_consumer_loop(brokers='localhost:9092', topic="fan_regulator_data", group_id=None, reset=False):

    if reset_mode:
        logger.info("🔄 Recovery mode: No saved model found, starting fresh.")
        group_id = f"fan-consumer-{int(time.time())}"
    else:
        logger.info("🔄 Live mode: Using existing model.")
        group_id = "fan_consumer_group"

    # group_id = group_id or f"fan-consumer-{int(time.time())}"
    
    consumer = KafkaFanConsumer(topic=topic, group_id=group_id, bootstrap_servers=brokers, reset=reset)
    consumer.modes.set_mode("manual")
    consumer.consume_messages()

# ================= Main: CLI helpers =================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument(
        "mode",
        nargs="?",  # makes it optional
        choices=["producer", "consumer"],
        default="consumer",  # default if nothing is passed
        help="Run producer or consumer (default: producer)"
    )
    p.add_argument("--brokers", default="localhost:9092", help="Kafka bootstrap servers")
    p.add_argument("--topic", default="fan_regulator_data", help="Kafka topic name")
    p.add_argument("--interval", type=float, default=2.0, help="Producer send interval (s)")
    p.add_argument("--group", default="fan_group", help="Consumer group id")
    # p.add_argument("--reset", action="store_true", help="Consumer: set auto.offset.reset=earliest")
    args = p.parse_args()

    if args.mode == "producer":
        run_producer_loop(brokers=args.brokers, topic=args.topic, interval=args.interval)
    else:
        reset_mode = not os.path.exists("fan_model.pt")
        run_consumer_loop(brokers=args.brokers, topic=args.topic, group_id=args.group, reset=reset_mode)

