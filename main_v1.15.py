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

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from paddleocr import PaddleOCR

from munkres import Munkres, print_matrix

import telebot
import io

plt.rcParams["figure.figsize"] = (20, 30)
plt.rcParams["figure.dpi"] = (150)


# ## Clases

# ### utility


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
            plt.text(r['xmin'], r['ymin'] + h // 2, r['name'][:5], color='green',
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


# ### finder


class Finder:
    def __init__(self, model) -> None:
        self.model = model

        self.ut = Utility()

        self.current_image = None
        self.intef_results = None
        self.table = None

    def open_image(self, filename, dir_in):
        # resize is helpfull
        self.current_image = Image.open(
            os.path.join(dir_in, filename)).convert('RGB')

    def interference(self):
        tiledImg = list(self.ut.tileImage(self.current_image))
        self.intef_results = self.model(tiledImg).pandas().xyxy
        return self.intef_results

    def form_table(self, threshold=0.0):
        w, h = self.current_image.size
        grid = list(self.ut.get_grid(h, w))
        self.table = pd.DataFrame(columns=self.intef_results[0].columns)

        for i in range(len(grid) - 1):
            res = self.intef_results[i].copy()
            res['xmin'] = res['xmin'] + grid[i][1]
            res['xmax'] = res['xmax'] + grid[i][1]
            res['ymin'] = res['ymin'] + grid[i][0]
            res['ymax'] = res['ymax'] + grid[i][0]
            self.table = self.table.append(res, ignore_index=True)
            self.table = self.table[self.table['confidence']
                                    > threshold].reset_index(drop=True)
        self.table = self.clear_table(self.table)
        return self.table

    def clear_table(self, table, iou_tresh=0.15):
        table['bbox'] = table.apply(lambda x: self.ut.add_bbox(x), axis=1)

        to_delete = list()
        to_concat = list()

        for i in range(table.shape[0]):
            for j in range(i + 1, table.shape[0]):
                pos, posj = table.iloc[i]['bbox'], table.iloc[j]['bbox']
                el, elj = table.iloc[i]['name'], table.iloc[j]['name']

                if self.ut.bb_iou(pos, posj) >= iou_tresh:
                    if self.ut.rating.index(el) < self.ut.rating.index(elj):
                        to_delete.append(j)
                        # print(f'Оставиили - {el}, убрали - {elj}')
                    elif self.ut.rating.index(el) > self.ut.rating.index(elj):
                        to_delete.append(i)
                        # print(f'Оставиили - {elj}, убрали - {el}')
                    else:
                        # print(f'Соеденили {elj}, {el}')
                        to_delete.append(j)
                        to_concat.append((i, j))

        return table.drop(index=to_delete).reset_index()


# ### reader


class Reader:
    def __init__(self) -> None:
        self.price = pd.read_excel('/Price-list-CHINT-ot-01_08_2022.xlsx',
                                   sheet_name='Тариф 01.08.2022', header=2)
        self.desc = self.price['Описание'].map(str.lower)

        self.ocr = PaddleOCR(use_angle_cls=True, lang='en',
                             debug=False, show_log=False)

    def read_from_labels(self, table, image):
        for i, r in table[table.name == 'v'].iterrows():
            xmin = int(r['xmin'] - (r['xmax'] - r['xmin']) * 0.3)
            ymin = int(r['ymin'] - (r['ymax'] - r['ymin']) * 0.3)
            xmax = int(r['xmax'] + (r['xmax'] - r['xmin']) * 0.3)
            ymax = int(r['ymax'] + (r['ymax'] - r['ymin']) * 0.3)
            croped_img = np.asarray(image)[ymin:ymax, xmin:xmax]
            plt.imshow(croped_img)
            founded = self.ocr.ocr(croped_img)
            txts = [line[1][0] for line in founded]
            yield self.find_in_price(' '.join(txts))

    def read_label(self, bbox, image):
        # xmin ymin xmax ymax
        # print(bbox)
        xmin = int(bbox[0] - (bbox[2] - bbox[0]) * 0.3)
        ymin = int(bbox[1] - (bbox[3] - bbox[1]) * 0.3)
        xmax = int(bbox[2] + (bbox[2] - bbox[0]) * 0.3)
        ymax = int(bbox[3] + (bbox[3] - bbox[1]) * 0.3)
        croped_img = np.asarray(image)[ymin:ymax, xmin:xmax]
        founded = self.ocr.ocr(croped_img)
        txts = [line[1][0] for line in founded]
        # print(txts)
        return ' '.join(txts)

    def find_in_price(self, founded):
        return process.extract(founded.lower(), self.desc, limit=3, scorer=fuzz.ratio)


# ### matcher


class Matcher:
    def __init__(self, table) -> None:
        exception_list = list()  # TODO
        vs = list()
        es = list()
        for i, r in table.iterrows():
            if r['name'] == 'v':
                vs.append(self.calculate_dists(i, r, table))
            else:
                es.append(self.calculate_dists(i, r, table))

        m = Munkres()
        indexes = m.compute(vs)

        v_ind, e_ind = self.get_table_indexes(table)
        self.match = [(v_ind[row], e_ind[column]) for row, column in indexes]

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
        a = ((row1['xmax'] - row1['xmin']) // 2 + row1['xmin']) - \
            ((row2['xmax'] - row2['xmin']) // 2 + row2['xmin'])
        b = ((row1['ymax'] - row1['ymin']) // 2 + row1['ymin']) - \
            ((row2['ymax'] - row2['ymin']) // 2 + row2['ymin'])
        return round(sqrt(a ** 2 + b ** 2))


# ### endTableFormer


class EndTableFormer:
    def __init__(self, img, finder, reader) -> None:
        self.finder = finder
        self.finder.current_image = img

        print('find Started')
        self.finder.interference()
        print('find Done')
        self.finder.form_table()
        print('table Done')
        self.match = Matcher(self.finder.table).match
        print('match Done')
        self.rd = reader
        print('reader Done')

    def get_end_table(self):
        # f_index | confidence | class | name | readed_related_str | founded_in_price | match | price
        end = {'f_index': [], 'confidence': [], 'class': [],
               'name': [], 'readed_related_str': [],
               'founded_in_price': [], 'match': []}
        for i, r in self.finder.table.iterrows():
            m = self.get_matched(i)
            if m is not None and r['name'] != 'v':
                readed = self.rd.read_label(
                    tuple(m['bbox'])[0], self.finder.current_image)
                end['f_index'].append(i)
                end['confidence'].append(r['confidence'])
                end['class'].append(r['class'])
                end['name'].append(r['name'])
                end['readed_related_str'].append(readed)
                end['founded_in_price'].append(
                    self.rd.find_in_price(readed)[0][0])
                end['match'].append(m.index[0])
        end = pd.DataFrame(end)
        return end

    def get_matched(self, el_index):
        for m in self.match:
            if m[0] == el_index:
                return self.finder.table.iloc[[m[1]]]
            elif m[1] == el_index:
                return self.finder.table.iloc[[m[0]]]
        return None


model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='./yolov5/yolov5/runs/train/03_08_2022_2_from_03_08_2022/weights/best.pt')  # local model

TOKEN = ':TOKEN:'
tb = telebot.TeleBot(TOKEN)


@tb.message_handler(commands=["start"])
def start(message, res=False):
    tb.send_message(message.chat.id, 'Отправьте схему.')


@tb.message_handler(content_types=["photo", 'document'])
def handle_text(message):
    try:
        if message.document:
            file_info = tb.get_file(message.document.file_id)
        else:
            file_info = tb.get_file(
                message.photo[len(message.photo) - 1].file_id)

        downloaded_file = tb.download_file(file_info.file_path)
        img = Image.open(io.BytesIO(downloaded_file)).convert('RGB')

        tb.send_message(message.chat.id,
                        f'Сейчас-сейчас... \nТикет: {file_info}')

        find = Finder(model)
        etf = EndTableFormer(img, find, Reader())

        table = etf.get_end_table()
        table.to_excel(f'{message.chat.id}.xlsx')
        fl = open(f"{message.chat.id}.xlsx", "rb")
        tb.send_document(message.chat.id, fl)
        Utility.draw(None, find.current_image, find.table,
                     match=etf.match, name=f'{message.chat.id}.png')
        tb.send_photo(message.chat.id, Image.open(
            f'{message.chat.id}.png').convert('RGB'))
    except:
        tb.send_message(message.chat.id, 'Что-то пошло не так.')


tb.polling(none_stop=True, interval=0)
