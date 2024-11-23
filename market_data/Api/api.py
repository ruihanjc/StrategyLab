import requests
import json


class BaseApi:

    def __init__(self):
        self.__init__

    def get(self, ticker):
        response = requests.get("http://api.open-notify.org/astros" + ticker)
        return response

    def post(self,headers):
        payload = headers
        r = requests.post('https://httpbin.org/post', data=payload)
        r.status_code

    