import cv2
from deepface import DeepFace
from collections import deque
import subprocess
import other_params

# -----------------------
# CONFIG
# -----------------------
CONFIDENCE_THRESHOLD = 0.4
EMOTION_HISTORY_LEN = 10
WEATHER_API_KEY = "8626216750fcb3381f91b31c71b0862e" 
COUNTRY_CODE = "US"  # for holiday detection
OLLAMA_MODEL = "llama2"  # Ollama model name

# -----------------------
# EMOTION DETECTION
# -----------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

emotion_history = deque(maxlen=EMOTION_HISTORY_LEN)

def preprocess_frame(frame):
    """Enhance frame for better detection"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return enhanced

def get_smoothed_emotion():
    if len(emotion_history) == 0:
        return "neutral"
    counts = {}
    for e in emotion_history:
        counts[e] = counts.get(e, 0) + 1
    return max(counts, key=counts.get)


# -----------------------
# OLLAMA CALL
# -----------------------
def ask_ollama(prompt_text):
    """Calls Ollama CLI and returns output"""
    try:
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL, prompt_text],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print("Error calling Ollama:", e)
        return ""

# -----------------------
# MAIN LOOP
# -----------------------
city, lat, lon = other_params.get_location()
today_status, tomorrow_status, weekday_name = other_params.today_and_tomorrow_status()
weather_desc, temperature = other_params.get_weather()
voice_tone = "neutral"  # placeholder, can integrate mic analysis

print(f"Location: {city}, Today: {today_status}, Weather: {weather_desc}, Temp: {temperature}°C")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    enhanced_frame = preprocess_frame(frame)
    gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=7, minSize=(60,60))

    for (x, y, w, h) in faces:
        face_roi = enhanced_frame[y:y+h, x:x+w]
        if face_roi.shape[0] > 0 and face_roi.shape[1] > 0:
            face_roi = cv2.resize(face_roi, (224,224))
        try:
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, detector_backend='opencv', silent=True)
            dominant_emotion = result[0]['dominant_emotion']
            confidence = result[0]['emotion'][dominant_emotion]/100.0
            if confidence >= CONFIDENCE_THRESHOLD:
                emotion_history.append(dominant_emotion)
        except:
            continue

    smoothed_emotion = get_smoothed_emotion()
    cv2.putText(frame, f"Emotion: {smoothed_emotion}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0),2)
    cv2.imshow("Emotion Tracker", frame)

    # --- Aggregate prompt and get recommendation ---
    prompt = f"""
    You are a movie recommendation assistant. The user context is:
    - Location: {city} ({lat:.2f}, {lon:.2f})
    - Today: {today_status} ({weekday_name})
    - Tomorrow: {tomorrow_status}
    - Current weather: {weather_desc}, {temperature}°C
    - Current mood: {smoothed_emotion}
    - Voice tone: {voice_tone}
    Recommend 5 movie genres or types that the user is most likely to enjoy right now.
    """
    recommendations = ask_ollama(prompt)
    print("Movie recommendations:", recommendations)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
