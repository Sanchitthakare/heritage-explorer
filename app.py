from flask import Flask, render_template, request, redirect, url_for, session, Response
from google.cloud import vision, texttospeech
from vertexai.preview.generative_models import GenerativeModel, Content, Part
import wikipediaapi
import requests
import os
import io
import re

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Google Cloud Clients
vision_client = vision.ImageAnnotatorClient()
tts_client = texttospeech.TextToSpeechClient()
model = GenerativeModel("gemini-1.5-pro")
wiki = wikipediaapi.Wikipedia(user_agent='HeritageExplorer/1.0 (contact@email.com)', language='en')

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def identify_site(image_path):
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = vision_client.landmark_detection(image=image)
    landmarks = response.landmark_annotations
    return landmarks[0].description if landmarks else None

def fetch_wiki_data(site_name):
    page = wiki.page(site_name)
    return page.summary if page.exists() else "No data found."

def generate_story(data):
    prompt = f"Create a vivid, engaging story about this heritage site, weaving in its history, cultural significance, key events, and unique traditions: {data}"
    response = model.generate_content(prompt)
    return response.text

def generate_fun_facts(site_name, data):
    prompt = f"Generate 2 short, fascinating 'Did You Know?' facts about {site_name} based on: {data}"
    response = model.generate_content(prompt)
    text = response.text
    facts = re.findall(r'(?:^|\n)(?:\d+\.|\-|\•)\s*(.*)', text)
    if not facts:
        facts = text.split('\n')
    return [fact.strip() for fact in facts if fact.strip()][:3]

def get_media(site_name):
    youtube_api_key = os.environ.get("YOUTUBE_API_KEY")
    unsplash_access_key = os.environ.get("UNSPLASH_ACCESS_KEY")

    youtube_url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={site_name}+history&type=video&key={youtube_api_key}"
    response = requests.get(youtube_url).json()
    videos = [f"https://www.youtube.com/embed/{item['id']['videoId']}" for item in response.get('items', [])[:2]]

    unsplash_url = f"https://api.unsplash.com/search/photos?query={site_name}&client_id={unsplash_access_key}"
    response = requests.get(unsplash_url).json()
    images = [result['urls']['regular'] for result in response.get('results', [])[:4]]

    return videos, images

def get_coordinates(site_name):
    gmaps_api_key = os.environ.get("GMAPS_API_KEY")
    url = f"https://maps.googleapis.com/maps/api/geocode/json?address={site_name}&key={gmaps_api_key}"
    response = requests.get(url).json()
    if response['status'] == 'OK':
        loc = response['results'][0]['geometry']['location']
        return loc['lat'], loc['lng']
    return None, None

def get_nearby_sites(lat, lng):
    gmaps_api_key = os.environ.get("GMAPS_API_KEY")
    url = (
        f"https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        f"?location={lat},{lng}&radius=5000&type=tourist_attraction&key={gmaps_api_key}"
    )
    response = requests.get(url)
    if response.status_code != 200:
        print("Request failed:", response.text)
        return []
    data = response.json()
    if data.get('status') != 'OK':
        print("Google API Error:", data.get('status'))
        return []

    return [
        {
            'name': place['name'],
            'lat': place['geometry']['location']['lat'],
            'lng': place['geometry']['location']['lng']
        }
        for place in data.get('results', [])[:3]
    ]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'photo' not in request.files:
        return redirect(url_for('home'))
    file = request.files['photo']
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    site_name = identify_site(image_path)
    if not site_name:
        return "Site not recognized. Try again.", 400

    wiki_data = fetch_wiki_data(site_name)
    story = generate_story(wiki_data)
    fun_facts = generate_fun_facts(site_name, wiki_data)
    videos, images = get_media(site_name)
    lat, lng = get_coordinates(site_name)
    nearby_sites = get_nearby_sites(lat, lng) if lat and lng else []

    session['site'] = {
        'name': site_name,
        'story': story,
        'videos': videos,
        'images': images,
        'lat': lat,
        'lng': lng,
        'fun_facts': fun_facts,
        'nearby_sites': nearby_sites,
        'chat_history': [],
        'language': 'en-US'
    }
    return redirect(url_for('story'))

@app.route('/story')
def story():
    site = session.get('site', {})
    return render_template('story.html', site=site)

@app.route('/stream_audio')
def stream_audio():
    site = session.get('site', {})
    story = site.get('story')
    lang = site.get('language', 'en-US')

    if not story:
        return "No story available", 400

    synthesis_input = texttospeech.SynthesisInput(text=story)
    voice = texttospeech.VoiceSelectionParams(
        language_code=lang,
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = tts_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    audio_stream = io.BytesIO(response.audio_content)
    return Response(audio_stream, mimetype='audio/mpeg')

@app.route('/chat', methods=['POST'])
def chat():
    site = session.get('site', {})
    message = request.form['message']
    site['chat_history'].append({'role': 'user', 'content': message})

    context = f"You are an expert tour guide for {site['name']}. Use this info to answer: {site['story'][:1000]}... Fun facts: {', '.join(site['fun_facts'])}"
    contents = [Content(role='model', parts=[Part.from_text(context)])] + \
               [Content(role=msg['role'], parts=[Part.from_text(msg['content'])]) for msg in site['chat_history']]
    chat = model.start_chat(history=contents[:-1])
    response = chat.send_message(contents[-1].parts[0].text).text
    site['chat_history'].append({'role': 'model', 'content': response})
    session['site'] = site
    return response

@app.route('/tour')
def tour():
    site = session.get('site', {})
    story_sections = site['story'].split('\n\n')  
    current_index = session.get('tour_index', 0)
    if current_index >= len(story_sections):
        current_index = 0
    session['tour_index'] = current_index
    section = story_sections[current_index]
    image = site['images'][current_index % len(site['images'])] if site['images'] else None
    return render_template('tour.html', site=site, section=section, image=image, index=current_index, total=len(story_sections))

@app.route('/tour/next')
def tour_next():
    story_sections = session.get('site', {}).get('story', '').split('\n\n')
    current_index = session.get('tour_index', 0)
    if current_index < len(story_sections) - 1:
        session['tour_index'] = current_index + 1
    return redirect(url_for('tour'))

@app.route('/tour/prev')
def tour_prev():
    current_index = session.get('tour_index', 0)
    if current_index > 0:
        session['tour_index'] = current_index - 1
    return redirect(url_for('tour'))

@app.route('/change_language/<lang>')
def change_language(lang):
    site = session.get('site', {})
    if site:
        site['language'] = lang
        session['site'] = site
    return redirect(url_for('story'))

if __name__ == '__main__':
    app.run(debug=True)
