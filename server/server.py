import cherrypy
import json
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
networks_dir = os.path.join(current_dir, "..", "networks")

class WebServer:
    @cherrypy.expose
    def index(self):
        return open('index.html', 'r').read()

    @cherrypy.expose
    def networks_list(self):
        return json.dumps(os.listdir(networks_dir))

    @cherrypy.expose
    def load_network(self, id):
        path = os.path.join(networks_dir, id, 'learn_log.json')
        return open(path, 'r')


if __name__ == '__main__':
    cherrypy.quickstart(WebServer(), config="cherry.conf")