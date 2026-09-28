import json

with open('data/precomputed_stats.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

js_content = f"window.EMBEDDED_PRECOMPUTED_DATA = {json.dumps(data, indent=2)};"

with open('static/js/precomputed_bundle.js', 'w', encoding='utf-8') as f:
    f.write(js_content)

print("static/js/precomputed_bundle.js written successfully!")
