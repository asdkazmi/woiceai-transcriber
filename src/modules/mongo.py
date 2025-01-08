from pymongo import MongoClient
import datetime
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)
load_dotenv()

uri = os.getenv('DB_URI')
# Create a new client and connect to the server
client = MongoClient(uri)

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    logger.info("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    logger.info(e)

# Access the 'Woice' database and the 'chunks' collection
db = client['woice-ai-kb']
collection = db['document-kb']



def insert_audio(all_chunks, embeddings, kbId, s3_key,title):
    # Prepare documents for insertion
    audio_transcription = [{
        "type": "audio",
        "title": title,
        "audio_url": s3_key,
        "datetime": datetime.datetime.now(),
        "kbId": kbId,
        "metadata": {"chunkNo": i+ 1},
        "content": all_chunks[i],
        "bgeVectors": embeddings[i]
    } for i in range(len(all_chunks))]

    # Insert documents into the collection
    collection.insert_many(audio_transcription)
