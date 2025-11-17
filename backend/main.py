from fastapi import FastAPI, File, UploadFile
from keras.models import load_model
import numpy as np
from PIL import Image
import io 

app = FastAPI()

model = load_model("nhl_logo_classifier_model.keras")

class_names = ['Anaheim Ducks', 'Boston Bruins', 'Buffalo Sabres', 
               'Calgary Flames', 'Carolina Hurricanes', 'Chicago Blackhawks', 
               'Colorado Avalanche', 'Columbus Blue Jackets', 'Dallas Stars',
               'Detroit Red Wings', 'Edmonton Oilers', 'Florida Panthers',
               'Los Angeles Kings', 'Minnesota Wild', 'Montreal Canadiens',
               'Nashville Predators', 'New Jersey Devils', 'New York Islanders',
               'New York Rangers', 'Ottawa Senators', 'Philadelphia Flyers',
               'Pittsburgh Penguins', 'San Jose Sharks', 'Seattle Kraken',
               'St. Louis Blues', 'Tampa Bay Lightning', 'Toronto Maple Leafs', 'Utah Mammoth',
               'Vancouver Canucks', 'Vegas Golden Knights', 'Washington Capitals',
               'Winnipeg Jets'] # 32 nhl teams 

@app.get("/")
def read_root():
    return {"message": "NHL Logo Classifier API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    
    predictions = model.predict(img_array)
    predicted_class = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_class])
    
    return {
        "team": class_names[predicted_class],
        "confidence": f"{confidence:.2%}",
        "class_id": predicted_class
    }


