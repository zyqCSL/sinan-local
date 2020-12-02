import sys
sys.path.append('../gen-py')

import uuid
from social_network import TextFilterService

from thrift import Thrift
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

def main():
  socket = TSocket.TSocket(host='localhost', port=40000)
  transport = TTransport.TFramedTransport(socket)
  protocol = TBinaryProtocol.TBinaryProtocol(transport)
  client = TextFilterService.Client(protocol)

  req_id = uuid.uuid4().int & 0x7FFFFFFFFFFFFFFF
  transport.open()

  text = "Thrift sucks!!!"
  print(client.UploadText(req_id, text, {}))
  transport.close()

if __name__ == '__main__':
  try:
    main()
  except Thrift.TException as tx:
    print('%s' % tx.message)
