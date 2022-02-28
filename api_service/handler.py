"""
Handler for RATSQL API.
"""
import os
import json
import traceback
import torch
import tornado.web
import logging
from logging import handlers
from inferance import Model


# from inferance_span import Model

# cuda = torch.cuda.is_available()
# device = 'cuda' if cuda else 'cpu'
# print('using device:{}'.format(device))
# os.environ["CUDA_VISIBLE_DEVICES"] = device

# logging.basicConfig(filename='./log/log.log', level=logging.INFO)
# logger = logging.getLogger(__name__)

def set_logger(context, logLevel=logging.DEBUG, logPath="log.log"):
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False

    logger = logging.getLogger(context)
    logger.setLevel(logLevel)
    # filebeat log format
    formatter = logging.Formatter('%(asctime)s [%(name)-12s] %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    file_handler = handlers.TimedRotatingFileHandler(logPath, when="H", interval=1, backupCount=10,
                                                     encoding="utf-8")
    file_handler.setLevel(logLevel)
    file_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(file_handler)
    return logger


logger = set_logger("inference", logging.INFO, './log/log.log')


class ner_handler(tornado.web.RequestHandler):
    """ handler to rat sql model """
    model = Model("./api_service/custom.jsonnet")
    # model = Model("./localtest.jsonnet")

    def post(self):
        """ post body of text to sql result """
        try:
            request_body = json.loads(self.request.body)
            logger.info(f'In: {request_body}')
            model_result = self.model.inference(request_body)
            response_body = json.dumps(model_result, ensure_ascii=False).encode('utf8')
            logger.info(f'Out: {model_result}')
            self.write(response_body)
        except Exception:
            try:
                logger.error(self.request.body)
            except:
                pass
            logger.error(traceback.format_exc())
            self.write('')
