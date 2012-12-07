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
        return json.dumps(sorted(os.listdir(networks_dir)))

    @cherrypy.expose
    def load_network_log(self, id):
        path = os.path.join(networks_dir, id, 'learn_log.json')
        return open(path, 'r').read()

    @cherrypy.expose
    def load_network_desc(self, id):
        path = os.path.join(networks_dir, id, 'desc.json')
        return open(path, 'r').read()

    @cherrypy.expose
    def load_data_visualizer(self, id):
        path = os.path.join(networks_dir, id, 'desc.json')
        info = json.load(open(path, 'r'))
        vis_path = os.path.join(networks_dir, id, info["data_visualizer"]["js_file"])
        return open(vis_path, 'r').read()

    @cherrypy.expose
    def load_data_file(self, id):
        path = os.path.join(networks_dir, id, 'desc.json')
        info = json.load(open(path, 'r'))
        vis_path = os.path.join(networks_dir, id, info["data_file"])
        return open(vis_path, 'r').read()

if __name__ == '__main__':
    cherrypy.quickstart(WebServer(), config="cherry.conf")