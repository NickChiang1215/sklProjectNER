import tornado.ioloop
import tornado.web
import handler
import os


def make_app():
    return tornado.web.Application([
        (r"/ner/", handler.ner_handler)
    ])


if __name__ == "__main__":
    app = make_app()
    PORT = int(os.getenv("PORT", 5001))
    app.listen(5001)
    print(f'app start at: {PORT}')
    tornado.ioloop.IOLoop.current().start()
