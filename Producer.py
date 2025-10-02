import json
import time
import random
import logging
from datetime import datetime
from confluent_kafka import Producer

# Voice Recognition
import speech_recognition as sr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | Producer | %(message)s"
)
logger = logging.getLogger("Producer")

# ================= Helpers =================
def get_season(month):
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "summer"
    elif month in [6, 7, 8, 9]:
        return "monsoon"
    else:
        return "autumn"

def simulate_sensor_data(prev_fan_level, regulator_level, mode="manual"):
    indoor_temp = round(random.uniform(22, 32), 1)
    outdoor_temp = round(random.uniform(20, 38), 1)
    indoor_humidity = round(random.uniform(40, 70), 1)
    outdoor_humidity = round(random.uniform(30, 80), 1)

    now = datetime.now()
    hour_of_day = now.hour
    day_of_week = now.weekday()
    season = get_season(now.month)

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
        "previous_fan_level": prev_fan_level if prev_fan_level is not None else 2,
        "recent_adjustment_trend": trend,
        "temperature_diff": round(indoor_temp - outdoor_temp, 3),
        "humidity_index": round(indoor_humidity - outdoor_humidity, 3),
        "regulator_level": regulator_level,
        "mode": mode
    }
    return message

# ================= Voice Command =================
def capture_voice_command(prompt="Say something"):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    with microphone as source:
        logger.info(f"🎤 Listening... {prompt}")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio).lower()
        logger.info(f"🗣️ Recognized voice: {text}")
        return text
    except sr.UnknownValueError:
        logger.warning("⚠️ Could not understand voice.")
        return None
    except sr.RequestError as e:
        logger.error(f"⚠️ Voice recognition error: {e}")
        return None

def extract_fan_level(text):
    if text is None:
        return None
    for word in text.split():
        if word.isdigit():
            level = int(word)
            if 1 <= level <= 5:
                return level
    for i in range(1, 6):
        if f"level {i}" in text or f"fan {i}" in text:
            return i
    return None

def extract_mode(text):
    if text is None:
        return None
    if "manual" in text:
        return "manual"
    elif "voice" in text:
        return "voice"
    elif "quit" in text or "exit" in text or "stop" in text:
        return "quit"
    return None

# ================= Producer Loop =================
def run_producer_loop(brokers="localhost:9092", topic="fan_regulator_data", interval=2.0):
    conf = {"bootstrap.servers": brokers}
    producer = Producer(conf)
    prev_level = None

    logger.info("🚀 Starting Producer... (Modes: manual | voice | quit)")
    try:
        while True:
            # 🔹 Ask for mode (voice or keyboard)
            mode_input = input("Enter mode (manual/voice/quit or say it): ").strip().lower()

            if mode_input == "":
                # If empty, listen via voice
                text = capture_voice_command("Say 'manual', 'voice' or 'quit'")
                mode_choice = extract_mode(text)
            else:
                mode_choice = mode_input

            if mode_choice == "quit":
                break

            if mode_choice == "manual":
                try:
                    regulator_level = int(input("Enter Fan Regulator Level (1-5): ").strip())
                except ValueError:
                    logger.warning("⚠️ Invalid input. Defaulting to level 3.")
                    regulator_level = 3
                data = simulate_sensor_data(prev_level, regulator_level, mode="manual")
                prev_level = regulator_level

            elif mode_choice == "voice":
                text = capture_voice_command("Say 'Fan level 1 to 5'")
                regulator_level = extract_fan_level(text)
                if regulator_level is None:
                    logger.warning("⚠️ No valid fan level found.")
                    continue
                data = simulate_sensor_data(prev_level, regulator_level, mode="voice")
                prev_level = regulator_level

            else:
                logger.warning("⚠️ Invalid mode. Use 'manual', 'voice', or 'quit'.")
                continue

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

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Fan Regulator Producer")
    p.add_argument("--brokers", default="localhost:9092", help="Kafka bootstrap servers")
    p.add_argument("--topic", default="fan_regulator_data", help="Kafka topic name")
    p.add_argument("--interval", type=float, default=2.0, help="Send interval (s)")
    args = p.parse_args()
    run_producer_loop(brokers=args.brokers, topic=args.topic, interval=args.interval)

