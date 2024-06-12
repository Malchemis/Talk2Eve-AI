from db_handler import DBHandler

if __name__ == "__main__":
    db = DBHandler()
    data = db.find()
    for d in data:
        print(d)
