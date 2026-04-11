from flask import Flask, render_template, request, jsonify
from supabase import create_client
import os

app = Flask(__name__)

# Replace these with your Supabase credentials
SUPABASE_URL = "https://your-project-id.supabase.co"
SUPABASE_KEY = "your-anon-key"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.route('/')
def index():
    return render_template('index.html')

# ESP32 will send data here
@app.route('/api/data', methods=['POST'])
def receive_data():
    content = request.json
    # Insert data into Supabase
    data = supabase.table("readings").insert({
        "distance": content['distance'], 
        "intensity": content['intensity']
    }).execute()
    return jsonify({"status": "success"}), 200

# Dashboard will fetch data from here
@app.route('/api/stats')
def get_stats():
    # Fetch the latest 20 readings
    response = supabase.table("readings").select("*").order("id", desc=True).limit(20).execute()
    rows = response.data
    
    latest_val = rows[0]['intensity'] if rows else 0
    return jsonify({
        "latest_intensity": latest_val,
        "history": [r['intensity'] for r in rows][::-1]
    })

# Vercel requires this for serverless execution
if __name__ == '__main__':
    app.run()