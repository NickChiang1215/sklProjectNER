"""
Handler for RATSQL API.
"""
import os
import json
import traceback
import torch
import tornado.web
from inferance import Model
# from inferance_span import Model

# cuda = torch.cuda.is_available()
# device = 'cuda' if cuda else 'cpu'
# print('using device:{}'.format(device))
# os.environ["CUDA_VISIBLE_DEVICES"] = device

class ner_handler(tornado.web.RequestHandler):
    """ handler to rat sql model """
    model = Model("./api_service/custom.jsonnet")

    def post(self):
        """ post body of text to sql result """
        try:
            # print("self.request.body", self.request.body)
            request_body = json.loads(self.request.body)
            # response_body = {'_result': self.model.inference(request_body)}
            response_body = json.dumps(self.model.inference(request_body), ensure_ascii=False).encode('utf8')
            self.write(response_body)
        except Exception:
            print(traceback.format_exc())
            self.write('')