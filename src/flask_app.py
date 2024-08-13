import logging
import json

from flask import Flask
from flask_restful import Api, Resource, request

from app import transcribe_text_from_url

app = Flask(__name__)
api = Api(app)

# Configure the logging system
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class Transcribe(Resource):

    def post(self):
        return transcribe_text_from_url(json.loads(request.data), [])


# Add the resource to the API
api.add_resource(Transcribe, '/transcribe')

if __name__ == "__main__":
    app.run(debug=True)
