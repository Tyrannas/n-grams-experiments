import requests
import json
import time

all_urls = json.load(open('./tabs_url_folk.json'))
current = 0
buffer_count = 1


def main(start=0):
    global buffer_count
    print(buffer_count)
    for index in range(start, len(all_urls)):
        url = all_urls[index]
        print(f'{index}/{len(all_urls)}')
        if buffer_count % 3 == 0:
            time.sleep(3)
        try:
            scrap(url)
            global current
            current = index
            buffer_count += 1

        except Exception as e:
            print(e)
            time.sleep(60)
            main(current)


def scrap(url):
    print(url)
    try:
        res = requests.get(url).content
    except Exception:
        print(current)
    temp = res.decode().split('window.UGAPP.store.page = ')
    temp2 = temp[1].split('window.UGAPP.store.i18n = {};')[0].strip()[:-1]
    dict = json.loads(temp2)
    song_name = dict['data']['tab']['song_name']
    tab = dict['data']['tab_view']['wiki_tab']['content']
    with open(f'./tabs_folk/{song_name}.txt', 'w', encoding="utf-8") as f:
        f.write(tab)


main(current)
