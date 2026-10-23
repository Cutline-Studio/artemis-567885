import urllib.request
import math
import numpy as np

def stream_url_to_coordinates(url, sample_points=64):
    """
    Artemis 567885 Live Streaming Wrapper.
    Fetches raw streaming bytes from a public URL using standard HTTP protocols 
    and directly projects them into a continuous normalized coordinate vector [0, 1] 
    without any discrete tokenization, vocabulary lookups, or string decoding.
    """
    print(f"[Artemis CLF] Connecting to internet source: {url}")
    try:
        # Use HTTP Range Request or stream to grab immediate raw byte payload
        req = urllib.request.Request(url, headers={'User-Agent': 'Artemis-567885-CLF/1.0'})
        with urllib.request.urlopen(req, timeout=5) as response:
            raw_bytes = response.read(sample_points * 4) # Read a slice of bytes
    except Exception as e:
        print(f"[Error] Network fetch failed: {e}")
        print("[Fallback] Generating synthetic internet data stream...")
        raw_bytes = bytes([int(127 + 127 * math.sin(i * 0.1)) for i in range(sample_points)])

    # Convert arbitrary raw byte values directly to normalized continuous coordinates [0.0, 1.0]
    byte_array = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
    if len(byte_array) < sample_points:
        # Pad with continuous noise if packet short-streams
        byte_array = np.pad(byte_array, (0, sample_points - len(byte_array)), 'edge')
    else:
        byte_array = byte_array[:sample_points]
        
    normalized_coordinates = byte_array / 255.0
    return normalized_coordinates

def fourier_feature_encoding(coordinates, mapping_size=32, scale=10.0):
    """
    Projects tokenless continuous coordinates into high-frequency Fourier spaces.
    """
    np.random.seed(567885)
    B = np.random.normal(size=(mapping_size, len(coordinates))) * scale
    
    # Project and compute continuous phase embeddings
    projection = np.dot(B, coordinates)
    fourier_features = np.concatenate([np.sin(projection), np.cos(projection)])
    return fourier_features

if __name__ == '__main__':
    # Test target website (Public metadata endpoint)
    target_url = "https://www.w3.org/TR/PNG/iso_8859-1.txt"
    
    # Execute the non-token internet streaming pipeline
    coords = stream_url_to_coordinates(target_url, sample_points=32)
    features = fourier_feature_encoding(coords)
    
    print("\n" + "="*50)
    print(" ARTEMIS 567885 - LIVE INTERNET WAVEFRONT STREAM")
    print("="*50)
    print(f"Raw Coordinates [0.0 - 1.0] Shape: {coords.shape}")
    print(f"Sampled Wave Values (First 5): {coords[:5]}")
    print(f"Continuous Fourier Vector Shape : {features.shape}")
    print(f"Encoded Field Signal (First 5)  : {features[:5]}")
    print("="*50)
