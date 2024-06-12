import pymongo
from ai_package_for_com.mongoDB.constants import MONGO_URI, MONGO_DB, MONGO_COLLECTION


class DBHandler:
    def __init__(self):
        try:
            self.client = pymongo.MongoClient(MONGO_URI)
            self.db = self.client[MONGO_DB]
            self.collection = self.db[MONGO_COLLECTION]
        except Exception as e:
            print("Erreur lors de la connexion à la bdd: ", e)
            exit(1)

    def insert(self, data):
        self.collection.insert_one(data)
    
    def update(self, filter, field, data):
        query = {'$set': {field: data}}
        self.collection.update_one(filter, query)

    def createIndex(self, indexName):
        self.collection.create_index([(indexName, 1)], unique=True)

    def findOne(self, query, display=None):
        return self.collection.find_one(query, display)

    def find(self, query=None, display=None):
        return self.collection.find(query, display)
