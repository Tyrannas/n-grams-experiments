import requests
import json

# baseURl = 'https://www.ultimate-guitar.com/explore?capo[]=0&order=rating_desc&type[]=Tabs&tuning[]=1&page='
baseURL = 'https://www.ultimate-guitar.com/explore?capo[]=0&genres[]=666&live[]=0&order=rating_desc&tuning[]=1&type[]=Tabs&page='
all_urls = []

for pageId in range(13):
    print(pageId)
    res = requests.get(baseURL + str(pageId + 1)).content
    temp = res.decode().split('window.UGAPP.store.page = ')
    tab_list = json.loads(temp[1][:temp[1].index(';\n')])[
        'data']['data']['tabs']
    urls = [x['tab_url'] for x in tab_list]
    all_urls += urls

with open('tabs_url_folk.json', 'w') as f:
    json.dump(all_urls, f)
