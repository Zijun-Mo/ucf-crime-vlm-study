"""Expose a metadata-only HTTP index of remote videos; never proxy video bytes.

For rclone's read-only HTTP FUSE mount with VFS disk caching disabled. GET on a
video redirects to its pinned public source; HEAD uses immutable manifest size.
"""
import argparse,html,json
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=18743)
args=parser.parse_args();data=json.loads(Path('research/localization_dev_v2/candidates.json').read_text())
records={r['name']:r for r in data['records']}
class Handler(BaseHTTPRequestHandler):
    def do_HEAD(self):self.answer(True)
    def do_GET(self):self.answer(False)
    def answer(self,head):
        name=unquote(self.path.lstrip('/'))
        if not name:
            body=('\n'.join('<a href="'+html.escape(n)+'">'+html.escape(n)+'</a>' for n in records)).encode()
            self.send_response(200);self.send_header('Content-Type','text/html');self.send_header('Content-Length',str(len(body)));self.end_headers()
            if not head:self.wfile.write(body)
        elif name in records:
            row=records[name]
            self.send_response(200 if head else 302)
            self.send_header('Content-Type','video/mp4');self.send_header('Content-Length',str(row['size']) if head else '0')
            self.send_header('Accept-Ranges','bytes')
            if not head:self.send_header('Location','https://huggingface.co/datasets/mteb/ucf-crime/resolve/'+data['dataset_revision']+'/'+row['repo_path'])
            self.end_headers()
        else:self.send_error(404)
ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
