"""Call the installed Codex desktop tool bridge for user-authorized events.

Wire format matches the installed codex-app-tools server.mjs NativePipeClient:
little-endian uint32 length followed by UTF-8 JSON-RPC, max 8 MiB.
"""
import json
import os
from pathlib import Path
import socket
import stat
import struct
import uuid


def call_desktop(tool,arguments,thread,event_id,pipe_path=None):
    candidates=[]
    preferred=pipe_path or os.environ.get('CODEX_APP_TOOLS_PIPE_PATH')
    if preferred:candidates.append(Path(preferred))
    folder=Path('/tmp')/f'codex-app-tools-{os.getuid()}'
    if folder.exists():candidates+=sorted(folder.glob('*.sock'),key=lambda p:p.stat().st_mtime,reverse=True)
    error=None
    for path in dict.fromkeys(candidates):
        call_sent=False
        try:
            info=path.stat()
            if info.st_uid!=os.getuid() or not stat.S_ISSOCK(info.st_mode):continue
            with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as channel:
                channel.settimeout(15);channel.connect(str(path))
                def receive(n):
                    chunks=[];left=n
                    while left:
                        data=channel.recv(left)
                        if not data:raise EOFError('Desktop bridge closed')
                        chunks.append(data);left-=len(data)
                    return b''.join(chunks)
                def rpc(method,params,ident):
                    nonlocal call_sent
                    body=json.dumps(dict(jsonrpc='2.0',id=ident,method=method,params=params)).encode()
                    if method=='tools/call':call_sent=True
                    channel.sendall(struct.pack('<I',len(body))+body)
                    size=struct.unpack('<I',receive(4))[0]
                    if size>8*1024**2:raise ValueError('Desktop bridge response exceeded limit')
                    result=json.loads(receive(size))
                    if 'error' in result:raise RuntimeError(result['error']['message'])
                    return result['result']
                catalog=rpc('tools/list',{'threadStartKind':'all'},1)
                spec=next(t for t in catalog['tools'] if t['name']==tool)
                request_id=uuid.uuid4().hex
                result=rpc('tools/call',dict(arguments=arguments,callerSource='codex',
                    callId='ml-monitor-'+event_id+'-'+request_id,namespace=spec['namespace'],threadId=thread,
                    tool=tool,turnId='ml-monitor-event-'+request_id),2)
                if not result.get('success'):
                    message=' '.join(x.get('text','') for x in result.get('contentItems',[]))
                    raise RuntimeError('Desktop tool rejected event: '+message[:2000])
                return result
        except (FileNotFoundError,ConnectionRefusedError,OSError) as exc:
            if call_sent:raise TimeoutError('Desktop event delivery acknowledgment uncertain') from exc
            error=exc
    raise RuntimeError('No reachable desktop tool bridge: '+str(error))
