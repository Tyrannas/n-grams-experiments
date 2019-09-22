def parse_tab(title):
    file = open(f'./tabs_folk/{title}').read()
    print(f"parsing {title}")

    import re
    r = re.compile('\|.+\|')
    # extract only the tab
    tabs = r.findall(file)

    # split each string into
    reformated = {}
    for index, tab in enumerate(tabs):
        if index % 6 not in reformated:
            reformated[index % 6] = []
        reformated[index % 6].append(tab)

    joined = [''.join(x).replace('|', '') for x in reformated.values()]
    joined = [''.join([word if word in ['0', '1', '2', '3', '4', '5', '6', '7',
                                        '8', '9', 'h', 'p'] else '-' for word in string]) for string in joined]

    # each item of the joined list is a guitar string
    # split every string in subelements
    # detect if a number has two digits or not
    # if yes merge all the subelements of the string with the next subelement
    cursor = 0
    words = []
    for index in range(len(joined[0])):
        take_two = False
        if cursor < index:
            for string in joined:
                numbers = [str(i) for i in list(range(10))]
                if string[cursor] in numbers and string[cursor + 1] in numbers:
                    take_two = True
            if take_two:
                word = ''.join([s[cursor:cursor+2] for s in joined])
                cursor += 2
            else:
                word = ''.join([s[cursor] for s in joined])
                cursor += 1
        elif cursor == index:
            word = ''.join([s[cursor] for s in joined])
        words.append(word)

    return words


def main():
    import os
    import json
    for tab in os.listdir('./tabs_folk'):
        try:
            words = parse_tab(tab)
        except Exception:
            continue
        tab = tab.split('.txt')[0]
        with open(f'./parsed_tabs_folk/{tab}.json', 'w') as fp:
            json.dump(words, fp)


main()
