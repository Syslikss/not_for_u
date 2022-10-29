class NewSelector:
    def __init__(self) -> None:
            self.ALIAS = {'motorized three-phase circuit breaker':  {'p|р': 3, 'ru_part': ['мото']},   'motorized single-phase circuit breaker':  {'p|р': 1, 'ru_part': ['мото']},   
                          'motorized switch':  {'ru_part': ['перек']},   'transformer':  {'ru_part': ['траснс']},   'motorizwd circuit breaker':  {'ru_part': ['мото']},   
                          'withdrawable circuit breaker': {},   'four-phase circuit breaker':  {'p|р': 4, 'ru_part': ['авт']},   'three-phase circuit breaker':  {'p|р': 3, 'ru_part': ['авт']},   
                          'three-phase switch':  {'p|р': 3, 'ru_part': ['перк']},   'two-phase circuit breaker':  {'p|р': 2, 'ru_part': ['пере']},   
                          'single-phase circuit breaker':  {'p|р': 1, 'ru_part': ['пере']},   'circuit breaker':  {'ru_part': ['пере']},   'surge protection device': {},   
                          'fuse switch disconnector': {},   'fuse': {},   'switch': {'ru_part': ['выкл']},   'capasitor ': {},   'photoresistor': {},   'direct connection counter': {},   
                          'voltage monitoring relay': {},   'lamp':  {'ru_part': 'ламп'},   'RCD 220V': {},   'differential circuit breaker 380V': {},   'differential circuit breaker 220V': {},   
                          'ground':  {'ru_part': ['зем']},   'instrument current transformer': {},   'Inductor': {},   'reactive power compensation device': {}}
            self.mods = ['a|а', 'ка|ka', 'mm|мм', 'p|р', 'гц|hz', 'квт|kw']

            self.price = self._prepare_prices('/content/drive/MyDrive/Price-list-CHINT-ot-01_08_2022.xlsx', 
                                              'Описание', sheet_name='Тариф 01.08.2022', header=2)
            self.most = self._prepare_prices('/content/drive/MyDrive/new_most_common.xlsx', 
                                             'Изделие', sheet_name='Sheet1', header=0)

    def _prepare_prices(self, file_path, desc_col, sheet_name=None, header=None):
        df = pd.read_excel(file_path, 
                                sheet_name=sheet_name, header=header)
        df['Описание'] = df[desc_col].apply(str.lower)
        df['info'] = df['Описание'].apply(self.get_score)
        return df

    def get_score(self, s):
        s = str(s)
        extracted = {}
        for m in self.mods:
            f = re.findall(r'(\b((?:\d*[,.]\d+|\d+))[|\s]?(' + m + '))', s)
            if len(f) > 0:
                extracted[f'{m}'] = f[0]
                s = s.replace(f[0][0], '')
        extracted['remain'] = s.strip()
        return extracted
    
    def compare_scores(self, x, y, ty):
        n = 0
        for k, v in self.ALIAS[ty].items():
            if k not in y.keys():
                y[k] = (None, str(v), None)
                
        if 'ru_part' in self.ALIAS[ty] and all(t in x['remain'] for t in self.ALIAS[ty]['ru_part']):
            n += 100

        for m in self.mods:
            if m in x.keys() and m in y.keys():
                if x[m][1] == y[m][1]:
                    n += 10
            else:
                n -= 1

        if x['remain'] != '' and y['remain'] != '':
            ratio = fuzz.partial_ratio(x['remain'], y['remain'])
            n += ratio // 10 

        return n

    def set_price(self, x):
        ar = self.get_score(str(x['readed_related_str']).lower())
        price_score = self.price['info'].apply(self.compare_scores, args=(ar, x['name'],))
        most_score = self.most['info'].apply(self.compare_scores, args=(ar, x['name'],))
        if price_score.max() > most_score.max() + 1:
            ret = self.price[price_score == price_score.max()][['Описание', 'Тариф с НДС, руб.']].iloc[0]
        else:
            ret = self.most[most_score == most_score.max()][['Описание', 'Сумма']].iloc[0]
        ret['ar'] = ar
        return ret

    def aug_with_prices(self, df):
        return pd.concat([df, df.apply(self.set_price, axis=1)], axis=1)
