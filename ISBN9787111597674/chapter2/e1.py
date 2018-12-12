import re

text_str = '文本最重要的来源无疑是网络。我们要把网络中的文本获取形成一个文本数据库。利用一个爬虫爬取到网络中的信息。爬取的策略有广度爬取和深度爬取。根据用户的需求，爬虫可以有主题爬虫和通用爬虫之分。'
p_string = text_str.split('。')

print('============ 爬虫 ============')
regex = '爬虫'
for line in p_string:
    if re.search(regex, line) is not None:
        print(line)

print('============ 爬. ============')
regex = '爬.'
for line in p_string:
    if re.search(regex, line) is not None:
        print(line)

print('============ ^文本 ============')
regex = '^文本'
for line in p_string:
    if re.search(regex, line) is not None:
        print(line)

print('============ 信息$ ============')
regex = '信息$'
for line in p_string:
    if re.search(regex, line) is not None:
        print(line)

text_str = ['[重要的]今年第七号台风23日登陆广东东部沿海地区', '上海发布车库销售监管通知：违规者暂停网签资格', '[紧要的]中国对印连发硬信息，印度急切需要结束对峙']
print('============ ^\[[重紧]..\] ============')
regex = '^\[[重紧]..\]'
for line in text_str:
    if re.search(regex, line) is not None:
        print(line)

print('============ \ ============')
if re.search('\\\\', 'I have one nee\dle') is not None:
    print('match it')
else:
    print('not match')

if re.search(r'\\', 'I have one nee\dle') is not None:
    print('match it')
else:
    print('not match')

print('============ years ============')
text_str = ['War of 1812', 'There are 5280 feet to a mail', 'Happy New Year 2016!']
years = []
for string in text_str:
    if re.search('[1-2][0-9]{3}', string):
        years.extend(re.findall('[1-2][0-9]{3}', string))
        print(string)

print('year: ' + str(years))
