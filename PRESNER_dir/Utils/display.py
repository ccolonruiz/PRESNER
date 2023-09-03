import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import tqdm
from IPython.display import display,HTML

label_names = ['DRUG', 'ROUTE', 'STRENGTH', 'FORM', 'DOSAGE']
color_dict = {label_names[0]:'rgba(255, 0, 0, 1)', 
              label_names[1]:'rgba(0, 255, 0, 1)', 
              label_names[2]:'rgba(255, 175, 0, 1)', 
              label_names[3]:'rgba(0, 0, 255, 1)', 
              label_names[4]:'rgba(255, 0, 200, 1)'}

"""
DISPLAY
"""

class html_span():
    def __init__(self, start, end, css={}):
        self.start = start
        self.end = end
        if '' in css: css.pop('')
        self.css = css
        
    def style(self):
        return "<span style='position:relative; display:inline-block; margin-bottom: 5px;white-space:pre;'>"
    
    def end_style(self):
        styles = "".join([
            f"<span style='position:absolute;display:inline-block;white-space:pre;left:0;right:0;top:{(v * 3) + 12}px;border-bottom:2px solid {color_dict[k[k.find(next(filter(lambda c: not c.isdigit(), k), '')):]]};'>"
            "</span>" for k, v in self.css.items()])
        return f"{styles}</span>"
    
class beauty_display():
    def __init__(self, data, spans, column, max_rows_to_display=10, n_jobs=-1):
        df = data.copy().drop(columns=['TEXT_drug_name', 'TEXT_quantity','chemblid'], errors='ignore')
        fk = df.fk if 'fk' in df.columns else df.index
        
        styles = dict(Parallel(n_jobs=n_jobs)(delayed(self.format_span)(k,spans[k]) for k in tqdm.tqdm(set(fk))))
        df['styles'] = fk.map(styles.get)
        
        df[column] = "<div style='min-width:250px;'>"+df.apply(lambda row: 
               ''.join([row[column][row['styles'][idx-1].end if idx else 0:hs.start]+hs.style()+
               row[column][hs.start:hs.end]+hs.end_style() for idx,hs in enumerate(row['styles'])] if len(row['styles']) else row[column]), 
               axis=1)+"</div>"
        
        df.drop('styles', axis=1,inplace=True)

        self.df = df
        self.max_rows_to_display = max_rows_to_display
    
    def format_span(self,key,span):
        
        def assign_heights(matrix):
            n = matrix.shape[0]
            heights = np.zeros(n, dtype=int)
            for i in range(n):
                for j in range(i + 1, n):
                    if matrix[j, i]:
                        if heights[i] == heights[j]:
                            heights[j] += 1
            return heights
        def map_html_spans(values):
            split_idx = (values[2:].shape[0]//2)+2
            return html_span(*values[:2], dict(zip(*(values[2:split_idx], values[split_idx:]))))
        
        aux_span = span.drop_duplicates().sort_values(by=['Start']).reset_index(drop=True)
        aux_span['Tag'] = aux_span.index.astype(str)+aux_span['Tag']
        endpoints, aux_index = np.union1d(span['Start'], span['End']), aux_span.index
        endpoints_T, aux_index_T = endpoints[:, np.newaxis], aux_index.values[:, np.newaxis]
        aux_start, aux_end = aux_span['Start'].values, aux_span['End'].values
        aux_start_T, aux_end_T = aux_start[:, np.newaxis], aux_end[:, np.newaxis]

        overlaps = np.transpose(aux_start <= aux_end_T) & (aux_index_T > aux_index)
        css_index = assign_heights(overlaps)

        masks=((aux_start < endpoints_T)[1:] * (aux_end > endpoints_T)[:-1])
        valid_mask_indices = np.any(masks, axis=1)

        _tags = (aux_span['Tag'].to_numpy() * masks)[valid_mask_indices]
        _css = ((css_index) * masks)[valid_mask_indices]
        _spans = np.vstack([endpoints[:-1][valid_mask_indices],endpoints[1:][valid_mask_indices]]).T

        html_spans = [map_html_spans(np.concatenate([_spans[i], _tags[i], _css[i]])) for i in range(_spans.shape[0])]
        return (key, html_spans)
    
    def _set_style(self, df, header=True):
        body = df.to_html(header=False,escape=False)
        if header:
            header = "<table class='dataframe'><thead><th></th>%s</thead>" % (
            ''.join(["<th style='color:%s;'> %s </th>" % (color_dict[c], c) if c in color_dict 
                     else "<th> %s </th>" % c for c in df.columns]))
            return header+body[body.find("<tbody>"):]
        return body
        
    def __getitem__(self, key):
        if isinstance(key, int):
            key = slice(key,key+1)
        display(HTML(self._set_style(self.df[key])))
            
    def __repr__(self):
        if self.df.shape[0] > self.max_rows_to_display:
            section = self.max_rows_to_display//2
            head = self._set_style(self.df.head(section))
            middle = f"<tr>{''.join(['<td>...</td>' for _ in range(len(self.df.columns)+1)])}</tr>"
            tail = self._set_style(self.df.tail(section))
            display(HTML(head[:head.rfind('</tbody>')]+middle+tail[tail.find("<tbody>"):]))
        else:
            display(HTML(self._set_style(self.df)))
        return ""
