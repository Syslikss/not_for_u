# -*- coding: utf-8 -*-
## Instals
"""

# # Commented out IPython magic to ensure Python compatibility.
# # %cd /content/drive/MyDrive/yolov5
# !pip install -r yolov5/requirements.txt  # install
# # %cd yolov5/
# 
# # !pip install fuzzywuzzy
# # !pip install python-Levenshtein
# 
# !pip install "paddleocr>=2.0.1" --upgrade # Recommend to use version 2.0.1+
# !pip install paddlepaddle
# 
# !pip install munkres
# 
# !pip install aiogram
# !pip install aiogram-utils

"""## Imports"""

from PIL import Image
from itertools import product

import os 
from math import ceil

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.pyplot import figure

import pandas as pd
import numpy as np

import torch

from math import sqrt

from paddleocr import PaddleOCR

from munkres import Munkres, print_matrix

# Convert2Pdf
import fitz
from typing import Tuple

import re

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

plt.rcParams["figure.figsize"] = (20, 30)
plt.rcParams["figure.dpi"] = (150)

import asyncio
from aiogram import Bot, Dispatcher, types
from datetime import datetime
from aiogram.types import InputFile
from aiogram.types import InlineKeyboardButton as ikb

"""## Clases

### utility
"""

class Utility:
    def __init__(self) -> None:
        self.rating = ['motorized three-phase circuit breaker',
                       'motorized single-phase circuit breaker',
                       'motorized switch', 'transformer', 'motorizwd circuit breaker',
                       'withdrawable circuit breaker', 'four-phase circuit breaker',
                       'three-phase circuit breaker', 'three-phase switch',
                       'two-phase circuit breaker', 'single-phase circuit breaker',
                       'circuit breaker', 'surge protection device',
                       'fuse switch disconnector', 'fuse', 'switch', 'capasitor ',
                       'photoresistor', 'direct connection counter',
                       'voltage monitoring relay', 'lamp', 'RCD 220V',
                       'differential circuit breaker 380V',
                       'differential circuit breaker 220V', 'ground',
                       'instrument current transformer', 'Inductor',
                       'reactive power compensation device', 'v', 'h']

        self.d = 600
        self.offset = 400

    def get_grid(self, h, w):
        def _get_grid(x): return range(0, self.d * ceil(x / self.d), self.d)
        return product(_get_grid(h), _get_grid(w))

    def tileImage(self, img):
        w, h = img.size
        grid = self.get_grid(h, w)

        for i, j in grid:
            box = (j, i, j + self.d + self.offset, i + self.d + self.offset)
            ni = img.crop(box)
            yield ni

    def calcBboxSquare(self, bbox):
        return abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

    def bb_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0

        boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

        return interArea / float(boxAArea + boxBArea - interArea)  # iou

    def add_bbox(self, x):
        return (x['xmin'], x['ymin'], x['xmax'], x['ymax'])

    def draw(self, img, table, name=None, match=None):
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_axis_off()
        for i, r in table.iterrows():
            w = r['xmax'] - r['xmin']
            h = r['ymax'] - r['ymin']
            bbox = patches.Rectangle((r['xmin'], r['ymin']), w, h, linewidth=1,
                                     edgecolor=(1, 0, 0), fill=False)
            plt.text(r['xmin'], r['ymin'] + h//2, r['name'][:5], color='green',
                     verticalalignment='bottom', size=6)
            ax.add_patch(bbox)
        if match:
            for m in match:
                r1 = table.iloc[[m[0]]]
                r2 = table.iloc[[m[1]]]
                plt.plot([r1['xmax'], r2['xmax']], [r1['ymax'],
                         r2['ymax']], color="blue", linewidth=2)

        if name:
            plt.savefig(name, format='png',
                        bbox_inches='tight', pad_inches=0.0)

    def open_image(file_path):
        #resize is helpfull
        return Image.open(os.path.join(file_path)).convert('RGB')

    def convert_pdf2img(file_path:str, pages:Tuple=None, scale=2):
        pdfIn = fitz.open(file_path)
        pixes = []
        for pg in range(pdfIn.pageCount):
            if str(pages) != str(None):
                if str(pg) not in str(pages):
                    continue
            page = pdfIn[pg]
            mat = fitz.Matrix(scale, scale)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pixes.append(Image.frombytes(
                "RGB", [pix.width, pix.height], pix.samples))
        pdfIn.close()
        return pixes

"""### finder"""

class Finder:
    def __init__(self) -> None:
        self.model = torch.hub.load('ultralytics/yolov5', 'custom',
                                    path='/content/drive/MyDrive/yolov5/yolov5/runs/train/03_08_2022_2_from_03_08_2022/weights/best.pt')
        self.ut = Utility()

    def interference(self, image):
        tiledImg = list(self.ut.tileImage(image))
        result = self.model(tiledImg).pandas().xyxy
        return result

"""### reader"""

class Reader:
    def __init__(self) -> None:
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en',
                             debug=False, show_log=False)

    def add_readed(self, table, image):
        table['readed'] = table.apply(
            self.read_from_labels, args=(image,), axis=1)
        return table

    def read_from_labels(self, r, image):
        if r['name'] == 'v':
            xmin = int(r['xmin'] - (r['xmax'] - r['xmin']) * 0.3)
            ymin = int(r['ymin'] - (r['ymax'] - r['ymin']) * 0.3)
            xmax = int(r['xmax'] + (r['xmax'] - r['xmin']) * 0.3)
            ymax = int(r['ymax'] + (r['ymax'] - r['ymin']) * 0.3)
            croped_img = np.asarray(image)[ymin:ymax, xmin:xmax]
            founded = self.ocr.ocr(croped_img)
            txts = [line[1][0] for line in founded[0]]
            return ' '.join(txts)
        else:
            return

"""### matcher"""

class Matcher:
    def __init__(self) -> None:
        self.m = Munkres()

    def get_matches(self, table):
        exception_list = list()  # TODO
        vs = list()
        es = list()
        for i, r in table.iterrows():
            if r['name'] == 'v':
                vs.append(self.calculate_dists(i, r, table))
            else:
                es.append(self.calculate_dists(i, r, table))

        indexes = self.m.compute(vs)

        v_ind, e_ind = self.get_table_indexes(table)
        match = [(v_ind[row], e_ind[column]) for row, column in indexes]
        return match

    def get_table_indexes(self, table):
        v_ind = list()
        e_ind = list()
        for i, r in table.iterrows():
            if r['name'] == 'v':
                v_ind.append(i)
            else:
                e_ind.append(i)
        return v_ind, e_ind

    def calculate_dists(self, index, row, table):
        res_d = list()
        for i, r in table.iterrows():
            if row['name'] == 'v':
                if r['name'] != 'v':
                    res_d.append(self.get_dist(row, r))
            else:
                if r['name'] == 'v':
                    res_d.append(self.get_dist(row, r))

        return res_d

    def get_dist(self, row1, row2):
        a = ((row1['xmax'] - row1['xmin'])//2 + row1['xmin']) - \
            ((row2['xmax'] - row2['xmin'])//2 + row2['xmin'])
        b = ((row1['ymax'] - row1['ymin'])//2 + row1['ymin']) - \
            ((row2['ymax'] - row2['ymin'])//2 + row2['ymin'])
        return round(sqrt(a**2 + b**2))

"""### table_former"""

class TableFormer:
    def __init__(self) -> None:
        self.ut = Utility()

    def form_table(self, size, intef_results, threshold=0.2):
        grid = list(self.ut.get_grid(size[1], size[0]))
        table = pd.DataFrame(columns=intef_results[0].columns)

        for i in range(len(grid)):
            res = intef_results[i].copy()
            res['xmin'] = res['xmin'] + grid[i][1]
            res['xmax'] = res['xmax'] + grid[i][1]
            res['ymin'] = res['ymin'] + grid[i][0]
            res['ymax'] = res['ymax'] + grid[i][0]
            table = table.append(res, ignore_index=True)

        table = table[table['confidence'] > threshold].reset_index(drop=True)
        table = self.clear_table(table)
        return table

    def clear_table(self, table, iou_tresh=0.15):
        table['bbox'] = table.apply(lambda x: self.ut.add_bbox(x), axis=1)
        rate = self.ut.rating
        to_delete, to_concat = list(), list()

        for i in range(table.shape[0]):
            for j in range(i + 1, table.shape[0]):
                pos, posj = table.iloc[i]['bbox'], table.iloc[j]['bbox']
                el, elj = table.iloc[i]['name'], table.iloc[j]['name']

                if self.ut.bb_iou(pos, posj) >= iou_tresh:
                    if rate.index(el) < rate.index(elj):
                        to_delete.append(j)
                    elif rate.index(el) > rate.index(elj):
                        to_delete.append(i)
                    else:
                        to_delete.append(j)
                        to_concat.append((i, j))

        return table.drop(index=to_delete).reset_index()

    def get_end_table(self, table, match):
        # f_index | confidence | class | name | readed_related_str | match | price
        end = {'confidence': [], 'name': [], 'readed_related_str': [], 'match': []}
        for i, r in table.iterrows():
            m = self.get_matched(i, table, match)
            if m is not None and r['name'] != 'v':
                end['confidence'].append(r['confidence'])
                end['name'].append(r['name'])
                end['readed_related_str'].append(m['readed'].tolist()[0])
                end['match'].append(m.index[0])
        end = pd.DataFrame(end)
        return end

    def get_matched(self, el_index, table, match):
        for m in match:
            if m[0] == el_index:
                return table.iloc[[m[1]]]
            elif m[1] == el_index:
                return table.iloc[[m[0]]]
        return None

"""### selector"""

class Selector:
    def __init__(self) -> None:
            self.ALIAS = {'motorized three-phase circuit breaker':  {'p|р': 3, 'ru_part': 'мото'},   'motorized single-phase circuit breaker':  {'p|р': 1, 'ru_part': 'мото'},   'motorized switch':  {'ru_part': 'перек'},   'transformer':  {'ru_part': 'траснс'},   'motorizwd circuit breaker':  {'ru_part': 'мото'},   'withdrawable circuit breaker': {},   'four-phase circuit breaker':  {'p|р': 4, 'ru_part': 'авт'},   'three-phase circuit breaker':  {'p|р': 3, 'ru_part': 'авт'},   'three-phase switch':  {'p|р': 3, 'ru_part': 'перк'},   'two-phase circuit breaker':  {'p|р': 2, 'ru_part': 'пере'},   'single-phase circuit breaker':  {'p|р': 1, 'ru_part': 'пере'},   'circuit breaker':  {'ru_part': 'пере'},   'surge protection device': {},   'fuse switch disconnector': {},   'fuse': {},   'switch': {},   'capasitor ': {},   'photoresistor': {},   'direct connection counter': {},   'voltage monitoring relay': {},   'lamp':  {'ru_part': 'ламп'},   'RCD 220V': {},   'differential circuit breaker 380V': {},   'differential circuit breaker 220V': {},   'ground':  {'ru_part': 'зем'},   'instrument current transformer': {},   'Inductor': {},   'reactive power compensation device': {}   }
            self.mods = ['a|а', 'ка|ka', 'mm|мм', 'p|р', 'гц|hz', 'квт|kw']

            self.price = pd.read_excel('/content/drive/MyDrive/Price-list-CHINT-ot-01_08_2022.xlsx',
                                    sheet_name='Тариф 01.08.2022', header=2)
            self.price['Описание'] = self.price['Описание'].apply(str.lower)
            self.price['info'] = self.price['Описание'].apply(self.get_score)

    def get_score(self, s):
        s = str(s)
        extracted = {}
        for m in self.mods:
            f = re.findall(r'(\b((?:\d*[,.]\d+|\d+))[|\s]?(' + m + '))', s)
            if len(f) > 0:
                extracted[f'{m}'] = f[0]
        extracted['ru_part'] = ' '.join(re.findall(r'[а-я]+', s))
        extracted['en_part'] = ' '.join(re.findall(r'[a-z]+', s))
        return extracted
    
    def compare_scores(self, x, y, ty):
        n = 0
        for k, v in self.ALIAS[ty].items():
            if k not in y.keys():
                y[k] = (None, str(v), None)
        if 'ru_part' in self.ALIAS[ty] and self.ALIAS[ty]['ru_part'] in x['ru_part']:
            n += 100
        for m in self.mods:
            if m in x.keys() and m in y.keys():
                if x[m][1] == y[m][1]:
                    n += 10
            else:
                n -= 1
        return n

    def set_price(self, x):
        ar = self.get_score(str(x['readed_related_str']).lower())
        r = self.price['info'].apply(self.compare_scores, args=(ar, x['name'],))
        ret = self.price[r == r.max()][['Описание', 'Тариф с НДС, руб.']].iloc[0]
        return ret

    def aug_with_prices(self, df):
        return pd.concat([df, df.apply(self.set_price, axis=1)], axis=1)

"""## Bot"""

FINDER = Finder()
TABLE_FORMER = TableFormer()
MATCHER = Matcher()
READER = Reader()
SELECTOR = Selector()

WORK_DIR = '/content/user_docs'

TOKEN = ':TOKEN:'


bot = Bot(token=TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=["start"])
async def cmd_start(m: types.Message):
    keyboard_markup = types.InlineKeyboardMarkup(resize_keyboard=True)
    keyboard_markup.row(ikb(text="KONERGY site", url="http://www.konergy.ru/"))
    await m.answer("Привет", reply_markup=keyboard_markup)
    with open(f'{WORK_DIR}/users.txt', 'a+') as fd:
        fd.write(f'{m.as_json()}\n')
        fd.close()


async def edit_inline(text, km, msg):
    km.row(ikb(text=text, callback_data='None'))
    await msg.edit_reply_markup(reply_markup=km)


@dp.message_handler(content_types=['document', 'photo'])
async def doc_photo_handler(m: types.Message):
    km = types.InlineKeyboardMarkup(resize_keyboard=True)
    km.row(ikb(text="Get File (1/3)", callback_data='None'))
    ms = await m.answer("Секунду...", reply_markup=km)

    inputs = []
    dir = f'{WORK_DIR}/{m.chat.id}'
    if m.content_type == 'photo':
        await m.photo[-1].download(destination_file=f'{dir}/input.png')
        inputs.append(Utility.open_image(f'{dir}/input.png'))
    elif m.content_type == 'document':
        doc = m.document
        file_info = await bot.get_file(doc.file_id)
        if doc.mime_subtype == 'pdf':
            await bot.download_file(file_info.file_path, f'{dir}/input.pdf')
            inputs.extend(Utility.convert_pdf2img(f'{dir}/input.pdf'))
        elif doc.mime_base == 'image':
            await bot.download_file(file_info.file_path, f'{dir}/input.png')
            inputs.append(Utility.open_image(f'{dir}/input.png'))

    for image in inputs:
        await edit_inline("Interference (2/3)", km, ms)
        intef_results = FINDER.interference(image)
        table = TABLE_FORMER.form_table(image.size, intef_results)

        match = MATCHER.get_matches(table)

        await edit_inline("Read Labels (3/3)", km, ms)
        READER.add_readed(table, image)
        end_table = TABLE_FORMER.get_end_table(table, match)
        aug_table = SELECTOR.aug_with_prices(end_table)

        await edit_inline("Done", km, ms)
        aug_table.to_excel(f'{dir}/output.xlsx')
        await m.reply_document(InputFile(f"{dir}/output.xlsx"))

        Utility.draw(None, image, table, match=match, name=f'{dir}/output.png')
        await m.reply_photo(InputFile(f'{dir}/output.png'))


async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

await main()

## TODO:
# 1. User: \/
#   a/ Register user \/ 
#   b/ User log \/
# 2. Selector
#   a/ Better 
# 3. Bot
#   a/ User (1) \/
#   b/ Inline button progression \/
#   c/ Buttons and guide 
#   d/ Refactor
#   e/ Stop by User
# 4. Excel
#   a/ something
# 5. PDF
#   a/ Select pages 
#

