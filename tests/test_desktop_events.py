import json,os,socket,struct,threading
from openjev_dss.desktop_events import call_desktop


def test_desktop_bridge_frames_and_routes_authorized_thread(tmp_path):
 path=tmp_path/'bridge.sock';server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);server.bind(str(path));server.listen(1);seen=[]
 def read(channel,n):
  b=b''
  while len(b)<n:b+=channel.recv(n-len(b))
  return b
 def serve():
  conn,_=server.accept()
  with conn:
   for _ in range(2):
    request=json.loads(read(conn,struct.unpack('<I',read(conn,4))[0]));seen.append(request)
    result=({'tools':[{'name':'send_message_to_thread','namespace':'codex_app'}]} if request['method']=='tools/list' else {'success':True,'contentItems':[{'type':'inputText','text':'accepted'}]})
    body=json.dumps({'jsonrpc':'2.0','id':request['id'],'result':result}).encode();wire=struct.pack('<I',len(body))+body
    conn.sendall(wire[:2]);conn.sendall(wire[2:7]);conn.sendall(wire[7:])
  server.close()
 thread=threading.Thread(target=serve);thread.start()
 result=call_desktop('send_message_to_thread',{'threadId':'authorized-thread','prompt':'event'},'authorized-thread','event-id',str(path))
 thread.join(timeout=3)
 assert result['success'] and seen[1]['params']['threadId']=='authorized-thread'
 assert seen[1]['params']['arguments']['threadId']=='authorized-thread'
 assert seen[1]['params']['tool']=='send_message_to_thread'
