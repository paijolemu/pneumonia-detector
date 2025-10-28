from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Kirim response sukses
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        # Buat pesan JSON sederhana
        message = {'status': 'success', 'message': 'Endpoint Python berjalan!'}
        self.wfile.write(json.dumps(message).encode('utf-8'))