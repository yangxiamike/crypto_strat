import json


json_dict = json.load(open('config.json', 'r'))

proxies = json_dict['proxies']
api_key = json_dict['api_key']
api_secret = json_dict['api_secret']
